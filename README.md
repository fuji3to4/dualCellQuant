# 🔬 DualCellQuant

A stepwise Gradio app for per-cell quantification with Cellpose-SAM segmentation and flexible masking. Two workflows are available via tabs in `dualCellQuant.py`:

- Dual images: Target + Reference images → AND mask → ratios and stats
- Single image: One image → (optional) Radial mask → Mask → stats

## Features ✨

- 🧠 Cellpose-SAM segmentation (lazy model load, optional GPU)
- 🌀 Independent, shape-aware Radial mask (optional)
- 🎯 Target/Reference masking with multiple strategies
  - none / global percentile / global Otsu / per-cell percentile / per-cell Otsu
  - Optional restriction to the Radial ROI (per label)
- 📊 Final integration and per-cell statistics
  - AND(Target, Reference[, Radial])
  - Mean/Sum per mask and per whole cell
  - Target/Reference ratio image and CSV export
- 🖼️ Clear overlays with cell IDs and downloadable NumPy arrays for all masks
- 🧹 Optional Preprocess: Background correction (Rolling ball) and Normalization (z-score, robust z, min-max, percentile)
  - Default: OFF
  - Preview processed images and download 8-bit TIFFs

## Workflows ▶️

### Single image (new)

1. Run Cellpose-SAM segmentation on the uploaded image
2. Build Radial mask (optional)
3. Apply Mask (none/global/per-cell; saturation limit; min object size)
4. Quantify per cell (on-mask and whole-cell stats; CSV export)

Outputs: label mask TIFF, mask TIFF, table CSV, image overlay previews.

### Dual images

1. 🧩 Run Cellpose-SAM segmentation

- (Optional) Preprocess
  - Background correction (Rolling ball, radius in px)
  - Normalization: z-score (default), robust z-score (median/MAD), min-max, percentile [1,99]
  - Preview and download preprocessed images (8-bit TIFF)


- Choose source image/channel and model thresholds
- Outputs: label mask (.npy), overlays

### 2. 🌀 (Optional) Build Radial mask

- Shape-aware ring between Inner% (0=center) and Outer% (100=boundary)
- Outer >100% expands outward into background only; labels are propagated via nearest cell
- Outputs: radial mask (bool .npy), radial labels (.npy), overlay

### 3. 🎯 Apply Target mask

- Choose channel, saturation limit, masking mode, percentile (if applicable), min object size
- Optionally restrict to the Radial ROI (per-label)
- Outputs: target mask (.npy), overlay
  - Note: Saturation limit thresholding is computed on the original image intensities (0–1 scale) so that preprocessing does not affect saturation gating.

### 4. 📎 Apply Reference mask

- Same options as Target; can also use Radial ROI
- Outputs: reference mask (.npy), overlay

### 5. 📊 Integrate & Quantify

- Optionally AND with Radial mask
- Produces per-cell table (CSV), AND mask (.npy), and a Target/Reference ratio image (.npy and preview)
  - If Preprocess is enabled, both visualization and measurement use the preprocessed arrays (for Dual/Single). For saturation gating in masking, the original image scale is used.

## UI Controls (by section) 🧰

- Segmentation params
  - Segment on: target/reference
  - Segmentation channel: 0 (gray/RGB→gray), 1 (R), 2 (G), 3 (B)
  - Diameter (px, 0=auto), Flow threshold, Cellprob threshold, Use GPU
- Radial mask (optional)
  - Radial inner % (0=center), Radial outer % (100=boundary, >100 expands outward)
  - Remove small objects (px)
  - Outputs: radial mask (bool), radial labels
  - Checkboxes: Use Radial ROI for Target/Reference mask
- Target mask
  - Channel, Masking mode, Saturation limit, Percentile, Remove small objects
- Reference mask
  - Same as Target
- Integrate
  - AND with radial mask (toggle)

- Preprocess (optional)
  - Background correction (Rolling ball): radius in px
  - Normalization method: z-score / robust z-score / min-max / percentile [1,99]
  - Preview buttons and TIFF download of preprocessed images

## Key Implementation Notes 🛠️

- File: `dualCellQuant.py`
- Model: `cellpose.models.CellposeModel(pretrained_model="cpsam")`
- Image handling: PIL↔NumPy helpers, robust grayscale extraction
- Masking modes:
  - Global percentile/Otsu: threshold computed on non-saturated pixels of the whole image
  - Per-cell percentile/Otsu: threshold computed from pixels inside each cell (and non-saturated)
  - Cleanup: `remove_small_objects` and a small `binary_opening`
- Radial mask internals:
  - Inside (≤100%): per-cell EDT normalized to [0..1] (center→boundary), band selection by percentage
  - Outside (>100%): background-only expansion; thickness scaled by each cell’s max internal distance; nearest-cell labeling via background EDT
- ROI-restricted masking:
  - Per-label intersection of Target/Reference mask with Radial mask
  - Cells without any pixels in the Radial mask get empty Target/Reference masks
- Integration and stats:
  - AND of selected masks; per-cell mean/sum for target/reference on AND mask and on full cell region
  - Ratio image limited to AND region, robustly normalized by 1–99 percentiles for visualization; NaN where reference==0

## Run locally 🚀

The app is a single-file Gradio UI. With Poetry:

```pwsh
poetry install
poetry run python .\dualCellQuant.py
```

Or with a plain environment (ensure Python 3.11+ and dependencies in `pyproject.toml`):

```pwsh
python .\dualCellQuant.py
```

Then open the local Gradio URL shown in the terminal.

Optionally, you can mount the app under FastAPI at `/dualcellquant`:

```pwsh
poetry run uvicorn serve:app --port 7860
```


## Outputs 📦

- Masks: `.npy` files for segmentation labels, target mask, reference mask, radial bool mask, radial label mask, AND mask
- CSV: per-cell summary table
- Ratio: Target/Reference as `.npy` (raw, with NaNs where ref==0) and an 8-bit preview image

## Known limitations / next steps ⚠️

- Radial outer >100% includes background near the cell. If you want Target/Reference masks computed only within ≤100%, add a toggle to clip ROI at 100%.
- No persistent project/session saving; use the exported `.npy`/`.csv` files to reproduce results.
- The Single image flow computes single-channel statistics only (no ratio). Ratios remain in the Dual images flow.
