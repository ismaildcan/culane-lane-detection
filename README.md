# Lane and Road-Boundary Detection under Adverse Visibility and Occlusion

A controlled comparison of **segmentation-based** and **curve-based** lane detection on a
curated [CULane](https://xingangpan.github.io/projects/CULane.html) subset, with
per-scenario error analysis on four test conditions: **Normal, Crowded, No-line, Night**.

> MSc Data Science · Machine Learning (PS26 – DSC02), Project 5 · Ismail Demircan
> Course submission — code only, per the submission instructions. The dataset is **not**
> redistributed here (see [Data](#data)).

## Overview

- **SEG** — a lightweight U-Net (ResNet18 encoder, ImageNet-pretrained, 14.3M params)
  predicts a 5-class lane mask; row-wise centroids yield lane points.
- **CURVE** — a 2nd-order polynomial `x = f(y)` fitted to the SEG points
  (interpolates occlusion gaps, smooths night noise).
- **Classical baseline** — a training-free bird's-eye colour+gradient detector,
  included to quantify what learning contributes.
- **Evaluation (geometric)** — official-style CULane **F1 @ IoU ≥ 0.5** (30 px line
  width, greedy matching) plus **mean lateral pixel error** on matched lanes.
  Validated end-to-end: ground-truth masks through the pipeline reproduce F1 = 1.0.
- **Loss** — recall-favoring **Focal + Tversky** (α=0.3, β=0.7 on the lane classes);
  replaced a background-dominated weighted CrossEntropy that silently under-segmented.

## Results (canonical Colab run, best checkpoint = epoch 6)

| Scenario | N | SEG F1 | CURVE F1 | Classical F1 | SEG px | CURVE px |
|---|---|---|---|---|---|---|
| Normal  | 567   | 0.361 | 0.357 | 0.104 | 10.1 | 9.5 |
| Crowded | 1,318 | 0.202 | 0.206 | 0.057 | 8.3  | 7.7 |
| No-line | 316   | 0.127 | 0.124 | 0.011 | 10.3 | 9.3 |
| Night   | 180   | 0.026 | 0.022 | 0.003 | 9.1  | 8.1 |

Key findings: the learned method beats the training-free baseline ~3.5× on Normal;
F1 degrades monotonically Normal → Crowded → No-line → Night (Night is the dominant
failure mode); CURVE matches SEG on detection but lowers lateral error by 0.6–1.0 px
in every scenario.

## Repository layout

```
.
├── culane_core.py                        # torch-free core: parsing, geometric F1,
│                                         # lateral metric, curve fit, classical baseline
├── training/
│   ├── lane_detection_culane.ipynb       # full Colab training pipeline (T4, ~90 min)
│   └── COLAB_README.md                   # how to run on Colab (zip → Drive → run)
└── presentation/
    └── lane_detection_presentation.ipynb # live-demo notebook: loads the saved
                                          # checkpoint, skips training, evaluates fast
```

> `culane_core.py` is also embedded in both notebooks via `%%writefile` (they are
> self-contained); the standalone copy is kept for convenient code review.

## Data

The CULane license does not permit redistribution, so **no data is included**.
Download from the [official CULane page](https://xingangpan.github.io/projects/CULane.html).
This project uses a documented subset (~6.1 GB): `driver_161_90frame` (train),
`driver_37_30frame` (test), both `laneseg_label_w16` mask packs, `list`, and the
corrected `annotations_new` overlay. Expected layout and split design are documented
inside the notebooks (Dataset / EDA sections).

**Splits (leakage-free):** train = driver_161, clip-based 90/10 train/val
(seed 42 → 16,373 / 1,860 frames); test = driver_37 (disjoint official partition),
2,381 frames across the four scenarios.

## Trained model

The canonical checkpoint (`unet_resnet18.pt`, ~57 MB, best-val epoch of the Colab T4
run) is not stored in the repository; it is shared alongside the submission (Google
Drive). Place it at `MyDrive/CULane/unet_resnet18.pt` and the presentation notebook
will load it and skip training.
Note: training is stochastic (hardware/precision-dependent); loading this shared
checkpoint reproduces all reported numbers exactly, since inference is deterministic.

## Quick start (Colab)

1. Upload the dataset zip to `MyDrive/CULane/robust-lanes-dataset.zip`
   (see `training/COLAB_README.md`).
2. To **reproduce training**: run `training/lane_detection_culane.ipynb` on a T4
   (~90 min; checkpoint saved to Drive).
3. To **evaluate / demo without training**: put the shared checkpoint at
   `MyDrive/CULane/unet_resnet18.pt` and run
   `presentation/lane_detection_presentation.ipynb` top to bottom.

## Requirements

`torch`, `segmentation_models_pytorch==0.3.4`, `albumentations==1.4.0`, `timm`,
`opencv-python`, `matplotlib`, `pandas` (install cells are included in the notebooks).

## Citation

If you use CULane, cite:

> X. Pan, J. Shi, P. Luo, X. Wang, X. Tang. "Spatial As Deep: Spatial CNN for
> Traffic Scene Understanding." AAAI 2018.

## Acknowledgements

Supervision: Prof. Dr. Iftikhar Ahmed. AI assistance (Anthropic Claude) was used for
code drafting, debugging and documentation; all methodological decisions and
verifications are the author's own.
