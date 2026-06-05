import math
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from os import path
import os


mu = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(args):
    args.mu = mu
    args.std = std
    valdir = path.join(args.data_dir, 'val')
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize(args.img_size),
                                                           transforms.CenterCrop(args.crop_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=args.mu, std=args.std)
                                                           ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
    return val_loader


# ----------------------------------------------------------------------------
# Attention-Fool helpers
# ----------------------------------------------------------------------------

def _l12_normalize(P):
    """L1,2 per-head normalization from Sec. 5 of the paper.

    P has shape [B, H, N, d_k]. ||P||_{1,2} = sum_i sqrt(sum_j P_ij^2). We
    divide by (1/n) * ||P||_{1,2} per (batch, head) so the average l2 row norm
    becomes 1, making B^{hl} commensurable across heads/layers.
    """
    n = P.size(-2)
    row_l2 = torch.linalg.norm(P, dim=-1)               # [B, H, N]
    denom = row_l2.sum(dim=-1, keepdim=True) / n        # [B, H, 1]
    denom = denom.clamp_min(1e-12).unsqueeze(-1)        # [B, H, 1, 1]
    return P / denom


def attn_fool_loss(qk_list, target_key_idx, target_query_idx=None,
                   layers=None, normalize=True):
    """Compute the Attention-Fool L_kq (or L_kq*) loss.

    qk_list: list over layers of (q, k) each shaped [B, H, N, d_k].
    target_key_idx: token index i* of the adversarial-patch key (int or [B] tensor).
    target_query_idx: if None, average over all queries (L_kq). Otherwise
        the query token index j* to target (L_kq*) — e.g. 0 for the CLS token.
    layers: iterable of layer indices to include; default = all layers.
    normalize: apply per-head L1,2 normalization of Q, K (paper default).

    Returns a scalar loss aggregated via smooth-max (logsumexp) over heads
    and layers, matching Sec. 5 of the paper.
    """
    if layers is None:
        layers = range(len(qk_list))

    per_layer = []
    for l in layers:
        q, k = qk_list[l]                               # [B, H, N, d_k]
        if normalize:
            q = _l12_normalize(q)
            k = _l12_normalize(k)
        d_k = q.size(-1)
        scale = 1.0 / math.sqrt(d_k)

        if target_query_idx is None:
            # L_kq^{hl} = mean_j B^{hl}_{j, i*}
            # Compute only the i*-th column of B per head: q @ k[:, :, i*, :]
            if torch.is_tensor(target_key_idx):
                idx = target_key_idx.view(-1, 1, 1, 1).expand(-1, k.size(1), 1, k.size(-1))
                k_star = torch.gather(k, 2, idx).squeeze(2)              # [B, H, d_k]
            else:
                k_star = k[:, :, target_key_idx, :]                       # [B, H, d_k]
            col = torch.einsum('bhnd,bhd->bhn', q, k_star) * scale        # [B, H, N]
            L_hl = col.mean(dim=-1)                                       # [B, H]
        else:
            # L_kq*^{hl} = B^{hl}_{j*, i*}
            if torch.is_tensor(target_key_idx):
                idx_k = target_key_idx.view(-1, 1, 1, 1).expand(-1, k.size(1), 1, k.size(-1))
                k_star = torch.gather(k, 2, idx_k).squeeze(2)             # [B, H, d_k]
            else:
                k_star = k[:, :, target_key_idx, :]
            if torch.is_tensor(target_query_idx):
                idx_q = target_query_idx.view(-1, 1, 1, 1).expand(-1, q.size(1), 1, q.size(-1))
                q_star = torch.gather(q, 2, idx_q).squeeze(2)
            else:
                q_star = q[:, :, target_query_idx, :]
            L_hl = (q_star * k_star).sum(dim=-1) * scale                  # [B, H]

        # smooth-max over heads: L_kq^l = logsumexp_h L_kq^{hl}
        per_layer.append(torch.logsumexp(L_hl, dim=-1))                   # [B]

    stacked = torch.stack(per_layer, dim=-1)                              # [B, L]
    # smooth-max over layers
    L_kq = torch.logsumexp(stacked, dim=-1)                               # [B]
    return L_kq.mean()


def cosine_step_size(alpha0, t, N):
    """Cosine-decayed PGD step size from Sec. 6.1 of the paper."""
    return alpha0 * 0.5 * (1.0 + math.cos(math.pi * t / max(N, 1)))


def normalized_momentum(prev_m, grad, beta=0.9):
    """Normalized momentum from Sec. 6.1: m^t = beta*m^{t-1} + (1-beta)*g/||g||_2."""
    flat = grad.reshape(grad.size(0), -1)
    norm = flat.norm(p=2, dim=1).clamp_min(1e-12).view(-1, *([1] * (grad.dim() - 1)))
    return beta * prev_m + (1.0 - beta) * (grad / norm)


def build_patch_mask(image_shape, patch_size, row, col, device):
    """Binary mask covering a [patch_size x patch_size] region at (row, col)."""
    B, C, H, W = image_shape
    mask = torch.zeros(1, 1, H, W, device=device)
    mask[:, :, row:row + patch_size, col:col + patch_size] = 1.0
    return mask


def apply_patch(x_norm, patch_01, mask, mu_t, std_t):
    """Composite a [0,1] adversarial patch into a normalized image.

    x_norm: normalized input image (mean/std already applied).
    patch_01: tensor in [0,1] with same shape as x_norm; only its values
        inside `mask` matter.
    mask: [1,1,H,W] binary mask.
    """
    patch_norm = (patch_01 - mu_t) / std_t
    return x_norm * (1.0 - mask) + patch_norm * mask


def patch_token_index(network, patch_row, patch_col, patch_size=16, img_size=224):
    """Token-sequence index of the patch occupying (patch_row, patch_col).

    Accounts for the leading CLS (and dist) tokens of ViT/DeiT variants.
    """
    grid = img_size // patch_size
    patch_idx = (patch_row // patch_size) * grid + (patch_col // patch_size)
    if 'distilled' in network.lower():
        return patch_idx + 2
    return patch_idx + 1


# ----------------------------------------------------------------------------
# Logging / metrics
# ----------------------------------------------------------------------------

class my_logger:
    def __init__(self, args):
        name = "{}_{}_{}_{}_{}.log".format(args.name, args.network, args.dataset, args.train_attack_iters,
                                           args.attack_learning_rate)
        args.name = name
        self.name = path.join(args.log_dir, name)
        with open(self.name, 'w') as F:
            print('\n'.join(['%s:%s' % item for item in args.__dict__.items() if item[0][0] != '_']), file=F)
            print('\n', file=F)

    def info(self, content):
        with open(self.name, 'a') as F:
            print(content)
            print(content, file=F)


class my_meter:
    def __init__(self):
        self.meter_list = {}

    def add_loss_acc(self, model_name, loss_dic, correct_num, batch_size):
        if model_name not in self.meter_list.keys():
            self.meter_list[model_name] = self.model_meter()
        sub_meter = self.meter_list[model_name]
        sub_meter.add_loss_acc(loss_dic, correct_num, batch_size)

    def clean_meter(self):
        for key in self.meter_list.keys():
            self.meter_list[key].clean_meter()

    def get_loss_acc_msg(self):
        msg = []
        for key in self.meter_list.keys():
            sub_meter = self.meter_list[key]
            sub_loss_bag = sub_meter.get_loss()
            loss_msg = ["{}: {:.4f}({:.4f})".format(x, sub_meter.last_loss[x], sub_loss_bag[x])
                        for x in sub_loss_bag.keys()]
            loss_msg = " ".join(loss_msg)
            msg.append("model:{} Loss:{} Acc:{:.4f}({:.4f})".format(
                key, loss_msg, sub_meter.last_acc, sub_meter.get_acc()))
        return "\n".join(msg)

    class model_meter:
        def __init__(self):
            self.loss_bag = {}
            self.acc = 0.
            self.count = 0
            self.last_loss = {}
            self.last_acc = 0.

        def add_loss_acc(self, loss_dic, correct_num, batch_size):
            for loss_name in loss_dic.keys():
                if loss_name not in self.loss_bag.keys():
                    self.loss_bag[loss_name] = 0.
                self.loss_bag[loss_name] += loss_dic[loss_name] * batch_size
            self.last_loss = loss_dic
            self.last_acc = correct_num / batch_size
            self.acc += correct_num
            self.count += batch_size

        def get_loss(self):
            return {x: self.loss_bag[x] / self.count for x in self.loss_bag.keys()}

        def get_acc(self):
            return self.acc / self.count

        def clean_meter(self):
            self.__init__()
