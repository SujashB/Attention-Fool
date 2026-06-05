"""Single-model AttnFool eval, runs in its own process so CUDA memory is fully
released on exit. Invoked by AttentionFool.ipynb via subprocess; prints a
single JSON line {"name", "clean_acc", "attnfool_acc"} on stdout."""
import argparse, json, sys, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from models.DeiT import (
    deit_tiny_patch16_224, deit_base_patch16_224, deit_base_patch16_384,
)
from models.vision_transformer import (
    vit_tiny_patch16_224, vit_base_patch16_224, vit_base_patch16_384,
)
from models.resnet import ResNet50
from utils import (
    apply_patch, attn_fool_loss, build_patch_mask, cosine_step_size,
    normalized_momentum, patch_token_index, mu as IMNET_MU, std as IMNET_STD,
)


def _resnet50():
    return ResNet50(pretrained=True), False, 224

MODEL_FACTORY = {
    'ResNet50'  : _resnet50,
    'ViT-T'     : lambda: (vit_tiny_patch16_224(pretrained=True), True, 224),
    'ViT-B'     : lambda: (vit_base_patch16_224(pretrained=True), True, 224),
    'ViT-B-384' : lambda: (vit_base_patch16_384(pretrained=True), True, 384),
    'DeiT-T'    : lambda: (deit_tiny_patch16_224(pretrained=True), True, 224),
    'DeiT-B'    : lambda: (deit_base_patch16_224(pretrained=True), True, 224),
    'DeiT-B-384': lambda: (deit_base_patch16_384(pretrained=True), True, 384),
}


def forward(model, x, is_transformer):
    if is_transformer:
        out, qk_list = model(x)
        return out, qk_list
    return model(x), None


def make_loader(img_size, batch_size, mean, std, crop_pct):
    resize = int(round(img_size / crop_pct))
    tfm = transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    ds = torchvision.datasets.ImageFolder(str(REPO / 'data' / 'imagenet_val'), transform=tfm)
    # Folder names are the true ImageNet class indices (e.g. "0002"), but ImageFolder
    # relabels them contiguously by sorted order. Remap back to the true index.
    idx_to_true = {idx: int(name) for name, idx in ds.class_to_idx.items()}
    ds.target_transform = idx_to_true.__getitem__
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=2, pin_memory=True), len(ds)


def attack_batch(model, is_transformer, X, y, network_name, cfg, img_size, mean, std):
    mu_t  = torch.tensor(mean).view(1, 3, 1, 1).to(X.device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(X.device)
    criterion = nn.CrossEntropyLoss()
    mask = build_patch_mask(X.shape, cfg['patch_size'], cfg['patch_row'], cfg['patch_col'], X.device)

    target_token_idx = (
        patch_token_index(network_name, cfg['patch_row'], cfg['patch_col'],
                          patch_size=cfg['patch_size'], img_size=img_size)
        if is_transformer else 0
    )

    x_01 = X * std_t + mu_t
    patch_01 = torch.rand_like(X) * mask + x_01 * (1 - mask)
    patch_01 = patch_01.detach().requires_grad_(True)
    m_state = torch.zeros_like(patch_01) if cfg['use_momentum'] else None

    for t in range(cfg['attack_iters']):
        if patch_01.grad is not None:
            patch_01.grad = None
        x_adv = apply_patch(X, patch_01, mask, mu_t, std_t)
        out, qk_list = forward(model, x_adv, is_transformer)
        loss = criterion(out, y)
        if is_transformer and cfg['attack_mode'] == 'AttnFool_kq':
            loss = loss + cfg['attn_w'] * attn_fool_loss(qk_list, target_token_idx, None)
        elif is_transformer and cfg['attack_mode'] == 'AttnFool_kqstar':
            loss = loss + cfg['attn_w'] * attn_fool_loss(qk_list, target_token_idx, 0)
        grad = torch.autograd.grad(loss, patch_01)[0]

        if cfg['use_momentum']:
            m_state = normalized_momentum(m_state, grad, beta=0.9)
            direction = m_state.sign()
        else:
            direction = grad.sign()

        alpha_t = cosine_step_size(cfg['attack_lr'], t, cfg['attack_iters'])
        with torch.no_grad():
            patch_01 = patch_01 + alpha_t * direction * mask
            patch_01 = patch_01.clamp(0, 1)
            patch_01 = patch_01 * mask + x_01 * (1 - mask)
        patch_01.requires_grad_(True)

    with torch.no_grad():
        x_adv = apply_patch(X, patch_01, mask, mu_t, std_t)
        out, _ = forward(model, x_adv, is_transformer)
    return out


@torch.no_grad()
def clean_eval(model, is_transformer, loader, device, max_imgs):
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out, _ = forward(model, X, is_transformer)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if total >= max_imgs:
            break
    return correct / total


def adv_eval(model, is_transformer, loader, name, img_size, cfg, device, max_imgs, mean, std):
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = attack_batch(model, is_transformer, X, y, name, cfg, img_size, mean, std)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if total >= max_imgs:
            break
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--num-imagenet', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--attack-iters', type=int, default=250)
    ap.add_argument('--attack-lr', type=float, default=8.0 / 255.0)
    ap.add_argument('--attack-mode', default='AttnFool_kq')
    ap.add_argument('--attn-w', type=float, default=1.0)
    ap.add_argument('--use-momentum', action='store_true')
    ap.add_argument('--patch-size', type=int, default=16)
    ap.add_argument('--patch-row', type=int, default=0)
    ap.add_argument('--patch-col', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = {
        'attack_iters': args.attack_iters, 'attack_lr': args.attack_lr,
        'attack_mode': args.attack_mode, 'attn_w': args.attn_w,
        'use_momentum': args.use_momentum, 'patch_size': args.patch_size,
        'patch_row': args.patch_row, 'patch_col': args.patch_col,
    }

    print(f'[{args.name}] device={device}', file=sys.stderr, flush=True)
    model, is_t, img_size = MODEL_FACTORY[args.name]()
    model = model.to(device).eval()

    # Use each model's own pretrained normalization/crop. ViT uses (0.5,0.5,0.5);
    # DeiT/ResNet use the ImageNet mean/std. A mismatch silently wrecks accuracy.
    if hasattr(model, 'model') and getattr(model.model, 'pretrained_cfg', None):
        pc = model.model.pretrained_cfg
        mean, std, crop_pct = list(pc['mean']), list(pc['std']), pc.get('crop_pct', 0.875)
    else:
        mean, std, crop_pct = IMNET_MU, IMNET_STD, 0.965

    loader, n = make_loader(img_size, args.batch_size, mean, std, crop_pct)
    print(f'[{args.name}] imagenet({img_size}): {n} images', file=sys.stderr, flush=True)

    t0 = time.time(); clean = clean_eval(model, is_t, loader, device, args.num_imagenet)
    print(f'[{args.name}] clean acc : {clean:.4f}  [{time.time()-t0:.1f}s]', file=sys.stderr, flush=True)
    t0 = time.time(); adv = adv_eval(model, is_t, loader, args.name, img_size, cfg, device, args.num_imagenet, mean, std)
    print(f'[{args.name}] attnfool  : {adv:.4f}  [{time.time()-t0:.1f}s]', file=sys.stderr, flush=True)

    print(json.dumps({'name': args.name, 'clean_acc': clean, 'attnfool_acc': adv}))


if __name__ == '__main__':
    main()
