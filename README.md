# Attention-Fool — Reproduction

A from-scratch reproduction of the adversarial-patch attack from:

> **Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness**
> Giulio Lovisotto, Nicole Finnie, Mauricio Munoz, Chaithanya Kumar Mummadi, Jan Hendrik Metzen — CVPR 2022.
> (`AttnFool.pdf` in this repo.)

The attack places a single adversarial patch in one ViT/DeiT token slot and optimizes it to hijack
**dot-product attention**. On top of the standard cross-entropy PGD objective (`Lce`) it adds an
**Attention-Fool** term that directly drives queries toward the patch's key:

- **`Lkq`** — pull *all* queries toward the patch key (averaged over queries).
- **`Lkq*`** — pull the *class token's* query toward the patch key (architecture-specific variant).

All work lives in [`AttentionFool.ipynb`](AttentionFool.ipynb) (orchestration) and
[`run_model.py`](run_model.py) / [`utils.py`](utils.py) (attack + losses). Raw numbers are in
[`results.json`](results.json).

## Setup

| | Value |
|---|---|
| Eval set | 1000 ImageNet-val images (`data/imagenet_val`) |
| Patch | 16×16, top-left corner (one token slot) |
| Optimizer | PGD, 250 iters, step `α = 8/255`, cosine-decayed |
| Momentum | normalized momentum, `β = 0.9` (optional) |
| Models | ResNet50, ViT-T/B, ViT-B-384, DeiT-T/B (timm pretrained) |
| timm | **1.0.27** |

Metric is **robust accuracy (%)** of the victim model under attack — **lower = stronger attack**.

## Results: ours vs. paper (Table 3)

Robust accuracy %, lower is better. `Δ` = ours − paper (positive = our attack is *weaker*).

### Clean accuracy (baseline, no attack)

| | ResNet50 | ViT-T | ViT-B | ViT-B-384 | DeiT-T | DeiT-B |
|---|---|---|---|---|---|---|
| Paper | 80.6 | 73.5 | 85.0 | 86.4 | 69.4 | 82.0 |
| Ours  | 79.6 | 73.5 | 83.8 | 83.8\* | 71.0 | 81.6 |

\* ViT-B-384 clean acc in `results.json` mirrors ViT-B and looks like a placeholder — re-measure if it matters.

### No momentum

| Config | Model | Paper | Ours | Δ |
|---|---|---|---|---|
| `Lce`        | ResNet50 | 55.1 | 63.5 | +8.4 |
|              | ViT-T    | 0.1  | 12.5 | +12.4 |
|              | **ViT-B**    | **13.5** | **50.5** | **+37.0** |
|              | ViT-B-384| 31.2 | 54.4 | +23.2 |
|              | DeiT-T   | 19.8 | 20.3 | +0.5 |
|              | DeiT-B   | 36.0 | 38.0 | +2.0 |
| `+Lkq`       | ViT-T    | 0.5  | 16.3 | +15.8 |
|              | ViT-B    | 5.0  | 46.0 | +41.0 |
|              | DeiT-T   | 13.1 | 18.5 | +5.4 |
|              | DeiT-B   | 35.5 | 35.2 | −0.3 |
| `+Lkq*`      | ViT-T    | 0.3  | 12.1 | +11.8 |
|              | ViT-B    | 2.6  | 48.3 | +45.7 |
|              | DeiT-T   | 11.7 | 18.2 | +6.5 |
|              | DeiT-B   | 33.7 | 35.1 | +1.4 |

### With momentum

| Config | Model | Paper | Ours | Δ |
|---|---|---|---|---|
| `Lce+Mom`    | ResNet50 | 49.0 | 59.3 | +10.3 |
|              | ViT-T    | 0.0  | 2.3  | +2.3 |
|              | ViT-B    | 3.1  | 38.1 | +35.0 |
|              | DeiT-T   | 1.5  | 1.0  | −0.5 |
|              | DeiT-B   | 16.8 | 16.6 | −0.2 |
| `+Mom+Lkq`   | ViT-T    | 0.0  | 4.3  | +4.3 |
|              | DeiT-T   | 0.0  | 0.2  | +0.2 |
|              | DeiT-B   | 19.3 | 13.7 | −5.6 |
| `+Mom+Lkq*`  | ViT-T    | 0.0  | 2.6  | +2.6 |
|              | ViT-B    | 0.1  | 32.9 | +32.8 |
|              | DeiT-T   | 0.0  | 0.4  | +0.4 |
|              | DeiT-B   | 13.1 | 14.3 | +1.2 |

## Analysis — where we match and where we don't

**DeiT reproduces well; ViT does not.** DeiT-T/B land within ~2–6 points of the paper across every
configuration. The whole **ViT family is under-attacked**, and it scales with model size
(ViT-T < ViT-B-384 < ViT-B), with **ViT-B off by ~37 points**.

**It is not the Attention-Fool loss.** The gap is already present in the plain `Lce` PGD baseline
(ViT-B 50.5 vs 13.5). Cross-entropy PGD has no attention term, so the `Lkq`/`Lkq*` implementation is
not the root cause — the underlying patch-PGD is simply too weak on ViT. The attention losses *do*
help (ViT-B 50.5 → 46.0 → 48.3), just from a broken baseline.

**The model ordering flipped.** In the paper, ViT-B (13.5) is *far easier* to attack than DeiT-B
(36.0). In our runs ViT-B (50.5) is *harder* than DeiT-B (38.0). A pure robustness difference would
preserve the ordering — the flip indicates the attack is *differentially weak on ViT specifically*.

### Why DeiT is stable across timm versions but ViT is not

This is the crux of the reproducibility gap:

- **DeiT** → timm exposes exactly **one** checkpoint per model: `fb_in1k`, Facebook's original DeiT
  release. Every timm version maps `deit_*_patch16_224` to the *identical* weights. → DeiT matches
  the paper regardless of version.
- **ViT** → timm exposes **11** checkpoints for `vit_base_patch16_224` (orig, augreg, **augreg2**,
  sam, miil, dino, mae, …) and the **default has been silently revised**. The 2022 paper-era default
  was `augreg_in21k_ft_in1k`; timm 1.0.27 now resolves the same name to **`augreg2_in21k_ft_in1k`**
  — a later re-finetune with different augmentation and different robustness.

Upgrading timm swapped the ViT weights out from under the experiment while leaving DeiT untouched.

### Checkpoint clean-acc triangulation (measured on our 1000 images)

| ViT-B checkpoint | our subset | ≈ full-val top-1 |
|---|---|---|
| `orig_in21k_ft_in1k` | 80.0% | ~81.8 |
| `augreg_in21k_ft_in1k` | 82.8% | ~84.5 |
| **`augreg2_in21k_ft_in1k`** (current) | **83.8%** | **~85.1** |

The paper's ViT-B clean = 85.0% matches the **AugReg** family — so the checkpoint difference
(`augreg` → `augreg2`) is real but, on its own, unlikely to explain a 37-point robustness gap. The
ordering flip points to an additional **ViT-specific attack/gradient issue** (candidate:
`QKWrapper` in `models/vision_transformer.py`) layered on top of the checkpoint swap.

## Known discrepancies / TODO

- [ ] Pin the ViT-B checkpoint to the paper-era `augreg_in21k_ft_in1k` and re-run `Lce` to isolate
      checkpoint effect from attack-code effect.
- [ ] Audit `QKWrapper` gradient flow on ViT (clean acc is correct, so it's an attack-path issue).
- [ ] Confirm whether the paper used constant vs. cosine-decayed `α` (we use cosine).
- [ ] Re-measure ViT-B-384 clean accuracy (current value looks like a placeholder).
- [ ] Fill remaining empty cells (`+Mom+Lkq` for ViT-B, ResNet50 attention rows N/A — no attention).

## Citation

```bibtex
@inproceedings{lovisotto2022attention,
  title={Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness},
  author={Lovisotto, Giulio and Finnie, Nicole and Munoz, Mauricio and Mummadi, Chaithanya Kumar and Metzen, Jan Hendrik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
