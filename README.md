# cloud_feedbacks

Constrain real-world cloud feedbacks using Perturbed Parameter Ensembles (PPEs) and CERES observations, following the approach of He & Soden (2016): a model's present-day cloud radiative effect (CRE) carries information about its cloud feedback under warming.

## Science overview

**Target**: δNetCRE = Δ(SW CRE + LW CRE) under patterned +4K SST forcing (W/m²). Divided by 4 gives the feedback parameter in W/m²/K.

**Input features**: Annual-mean SW CRE and LW CRE maps on the HadGEM native 144×192 grid (1.25°×1.875°).

**Strategy**: Train on a large PPE (HadGEM GA8, 503 members), evaluate on a withheld PPE generation (GA9), a structurally different PPE (CESM2), and structurally diverse CMIP6 models (CFMIP). Apply to CERES observations for the real-world constraint.

### CRE definitions
```
SW CRE = rsutcs − rsut          (positive = cooling)
LW CRE = rlutcs − rlut          (positive = warming)
Net CRE = SW CRE + LW CRE
```

For CESM2, SW CRE is unavailable directly. The mean-state proxy is `FSNTOAC + FSUTOA` (= rsdt − SW_CRE). Delta SW CRE is exact because rsdt cancels in the difference. CESM2 SW therefore should not be used in cross-model evaluation.

---

## Datasets

| Dataset | Members | Forcing | Native grid | Notes |
|---|---|---|---|---|
| HadGEM GA8 PPE | 503 | amipFuture (+4K patterned) | 144×192 | Training set |
| HadGEM GA9 PPE | 503 | amipFuture (+4K patterned) | 144×192 | Same-family OOS test |
| CESM2 PPE | 262 | Uniform +4K | 192×288 | Cross-model OOS test (LW only) |
| CFMIP | 10 models | amip + amip-future4K | 2.5° (native varies) | Structural generalization test |
| CERES EBAF-TOA Ed4.2.1 | — | Observed | 1°→2.5° | Observational constraint |

CFMIP models used: BCC-CSM2-MR, CESM2, CNRM-CM6-1, GISS-E2-1-G, HadGEM3-GC31-LL, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0, NorESM2-LM. GFDL-CM4 excluded (no amip PD run on ESGF).

---

## Repository layout

```
cloud_feedbacks/
├── preprocess.py          # Stage 1: raw NetCDF → data/*.nc  (CRE + feedback files)
├── prepare_data.py        # Stage 1b: regrid all datasets to HadGEM 144×192 → data/tensors.npz
├── model.py               # CloudFeedbackCNN architecture
├── train_baseline.py      # Stage 2: ridge regression CV → checkpoints/baseline/
├── train_cnn.py           # Stage 3: CNN training → checkpoints/cnn/
├── sanity_checks.ipynb    # Figures: feedback histograms, CRE maps, z-score vs CERES
├── stage2_baseline.ipynb  # Figures: ridge regression evaluation (loads checkpoints)
├── stage3_cnn.ipynb       # Figures: CNN evaluation (loads checkpoints)
├── data/
│   ├── hadgem_ga8_cre.nc  hadgem_ga8_fb.nc
│   ├── hadgem_ga9_cre.nc  hadgem_ga9_fb.nc
│   ├── cesm2_cre.nc       cesm2_fb.nc
│   ├── cfmip_cre.nc       cfmip_fb.nc
│   ├── ceres_cre.nc
│   └── tensors.npz        # prepared 144×192 tensors (output of prepare_data.py)
└── checkpoints/
    ├── baseline/          # sklearn models + predictions
    └── cnn/               # .pt checkpoints + norm_stats.npz + predictions
```

---

## Run order

```bash
# 1. Preprocess raw data (run once per dataset update)
python preprocess.py

# 2. Regrid everything to HadGEM 144×192 and save tensors (run once)
python prepare_data.py

# 3. Ridge regression baseline
python train_baseline.py

# 4. CNN training
python train_cnn.py

# 5. Open notebooks for figures/evaluation
jupyter notebook stage2_baseline.ipynb
jupyter notebook stage3_cnn.ipynb
```

---

## Model architecture

`CloudFeedbackCNN` in [model.py](model.py):

```
Input (B, 2, 144, 192)   — SW CRE and LW CRE, per-pixel normalised vs GA8 ensemble

ConvBlock(2→32,   k=3)   GeoPad → Conv2d → BN → Mish     RF = 3 px
MaxPool2d(2)             144×192 → 72×96
ConvBlock(32→64,  k=3)                                     RF = 7 px (on input grid)
MaxPool2d(2)             72×96   → 36×48
ConvBlock(64→128, k=3)                                     RF = 15 px
ConvBlock(128→256,k=3)                                     RF = 19 px (~24° lat)
AdaptiveAvgPool2d(1)     global summary → (B, 256)
Linear(256→128) → Mish → Dropout(0.2) → Linear(128→1)

Output (B,)              — δNetCRE in normalised units (denormalised after)
```

**GeoPad2d**: circular padding along longitude (no date-line seam), reflection padding along latitude (natural pole boundary). All Conv2d layers use `padding=0` — GeoPad is the only padding applied.

**Training**: AdamW + CosineAnnealingLR, early stopping on validation MSE (patience=25), 10-fold CV on GA8, final model trained on all GA8 with 10% held out for early stopping.

**Normalisation**: per-pixel z-score fit on GA8 ensemble mean/std, applied to all datasets including CERES.

---

## Baseline results (ridge regression, `checkpoints/baseline/`)

| Dataset | N | R² | RMSE |
|---|---|---|---|
| GA8 10-fold CV | 503 | 0.871 | 0.207 W/m² |
| GA9 out-of-sample | 503 | 0.404 | 0.569 W/m² |
| CESM2 LW only | 262 | −3.857 | 1.657 W/m² |
| CFMIP structural | 10 | −4.817 | 2.389 W/m² |

CERES constraint (linear baseline): **+3.81 W/m² (+0.95 W/m²/K)**

CESM2 (both channels) is intentionally excluded from the summary — the SW proxy makes it non-comparable. The large generalization gap (GA8 R²=0.87 → GA9 R²=0.40) is the primary motivation for the CNN.

---

## Open questions / next steps

- Does the CNN close the GA8→GA9 generalization gap?
- CFMIP structural test: does the CNN generalize across GCM families?
- Explainability: Integrated Gradients (Captum) to identify which CRE regions drive the prediction
- CESM2: consider LW-only CNN training track for a fair cross-model comparison
