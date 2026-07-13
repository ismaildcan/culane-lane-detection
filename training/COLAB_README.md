# Running the project on Colab — step by step

Everything runs on a free Colab T4 GPU. You only need to (1) push the data to
your Google Drive once, and (2) open the notebook in Colab.

---

## 1. Zip the dataset on your Mac (once)

Open **Terminal** and run:

```bash
cd ~/Desktop/robust-lanes
zip -r robust-lanes-dataset.zip dataset
```

This makes `robust-lanes-dataset.zip` (~6 GB) whose top-level folder is `dataset/`.

> Tip: the zip won't shrink much (JPGs are already compressed) — that's normal.

## 2. Upload the zip to Google Drive

Put it at exactly this path so the notebook finds it:

```
My Drive / CULane / robust-lanes-dataset.zip
```

(Create a `CULane` folder in Drive, drag the zip in. Free Drive = 15 GB, so 6 GB fits.)
Upload takes a while depending on your connection — let it finish.

## 3. Open the notebook in Colab

- Go to <https://colab.research.google.com> → **File → Upload notebook** →
  pick `colab/lane_detection_culane.ipynb`.
- **Runtime → Change runtime type → GPU (T4)**.
- Run cells top to bottom. The notebook will:
  1. install deps, mount Drive, **unzip the data to Colab local disk** (fast I/O),
  2. train the U-Net (~12 epochs; checkpoint saved back to Drive),
  3. evaluate SEG and CURVE methods per scenario (F1 + lateral-distance),
  4. run the classical CV baseline, and
  5. draw one qualitative example per scenario.

### If your GPU quota is tight
- Lower `CFG['epochs']` (e.g. 6), or
- subsample training: in the Dataset cell, slice `self.items = self.items[::2]`.
- The checkpoint is on Drive, so you can **skip training** on later runs and jump
  straight to Section 7+ (evaluation) by just loading the checkpoint.

---

## Project structure

```
robust-lanes/
├── colab/                          ← Colab runtime files (run from here)
│   ├── COLAB_README.md             ← this file (how to run on Colab)
│   ├── lane_detection_culane.ipynb ← the full Colab pipeline (self-contained)
│   └── culane_core.py              ← local reference copy of the core logic
├── dataset/                        ← data (stays at root; zipped & pushed to Drive)
└── docs/
    ├── DATA.md                     ← technical data reference
    └── dataset_choices_and_limitations.md  ← dataset choices, rationale & limitations
```

> Folders for run outputs (report, slides, checkpoints, metrics) will be added
> later once we know what the notebook produces.

> `dataset/` must stay at the project root — the zip command above
> (`zip -r robust-lanes-dataset.zip dataset`) and the notebook's unzip logic
> depend on it.

## What's what

| Path | Purpose |
|------|---------|
| `colab/lane_detection_culane.ipynb` | the full Colab pipeline (self-contained) |
| `colab/culane_core.py` | data loading + geometric metrics + curve method (also embedded in the notebook via `%%writefile`; kept here for local use/reference) |
| `docs/DATA.md` | technical data reference |
| `docs/dataset_choices_and_limitations.md` | dataset choices, rationale & limitations |

## Method summary (for the report)

- **Segmentation method:** U-Net + ResNet18 (ImageNet-pretrained), 5-class mask
  (background + 4 lanes) → row-wise centroids give lane points.
- **Curve method:** 2nd-order polynomial fit to those points — interpolates
  occlusion gaps and smooths night noise.
- **Evaluation (geometric, not pixel overlap):** official CULane F1 at IoU≥0.5
  with 30 px line width, **plus** mean lateral pixel error on matched lanes.
- **Baseline:** training-free bird's-eye classical CV detector (near-zero F1 on
  CULane — quantifies why a learned method is needed).

> Verified before delivery: split sizes (567/1318/316/180), the metric
> (GT-mask→points reproduces F1=1.0), and that the curve method cuts lateral
> error ~55% on noisy output. Training runs on Colab.
