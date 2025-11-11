"""
Core functionality for DualCellQuant.

Includes:
- Segmentation (Cellpose)
- Mask application
- Quantification
- Image preprocessing
"""

# Stepwise Cellpose-SAM Gradio App
"""
Stepwise pipeline for Cellpose-SAM with Gradio.

Stages:
1. Run Cellpose segmentation.
2. Build optional radial ROI.
3. Apply Target/Reference masks.
4. Integrate & Quantify (preprocess applied only here).

This allows parameter tuning at each stage before integration.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import tempfile
from skimage import filters, morphology, measure, color, restoration
import scipy.ndimage as ndi
from cellpose import models
import io
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Import visualization functions (needed for overlays)
from .visualization import (
    colorize_overlay,
    vivid_label_image,
    annotate_ids,
    save_bool_mask_tiff,
    save_label_tiff,
)

# -----------------------
# Display/label settings (adjustable via UI)
# -----------------------
LABEL_SCALE: float = 1.8

# -----------------------
# Model lazy loader
# -----------------------
_MODEL: Optional[models.CellposeModel] = None


def get_model(use_gpu: bool = False) -> models.CellposeModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    return _MODEL

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    # Keep original dtype info to decide scaling strategy
    orig = np.array(img)
    arr = arr.astype(np.float32, copy=False)
    if np.issubdtype(orig.dtype, np.integer):
        # Integer images: scale by dtype max into 0-1
        max_val = float(np.iinfo(orig.dtype).max)
        if max_val > 0:
            arr /= max_val
    elif np.issubdtype(orig.dtype, np.floating):
        # Float images: if values exceed 1.0, normalize by max to 0-1
        vmax = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if vmax > 1.0:
            arr = arr / vmax
    # Final clamp to [0,1]
    return np.clip(arr, 0.0, 1.0)

def pil_to_numpy_native(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.float32, copy=False)

def background_correction(
    arr: np.ndarray,
    radius: int,
    enabled: bool,
    *,
    mode: str = "rolling",
    dark_pct: float = 5.0,
    manual_value: float | None = None,
) -> np.ndarray:
    if not enabled:
        return arr
    mode = (mode or "rolling").strip().lower()

    def _dark_subtract_2d(ch: np.ndarray, pct: float) -> np.ndarray:
        p = float(np.clip(pct, 0.0, 50.0))
        thr = float(np.percentile(ch, p))
        dark = ch[ch <= thr]
        if dark.size == 0:
            return ch
        m = float(np.mean(dark))
        return np.clip(ch - m, 0.0, None)

    def _manual_subtract_2d(ch: np.ndarray, val: float | None) -> np.ndarray:
        if val is None or not np.isfinite(val):
            return ch
        return np.clip(ch - float(val), 0.0, None)

    if mode == "manual":
        if arr.ndim == 2:
            return _manual_subtract_2d(arr, manual_value)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            chs = []
            for i in range(3):
                chs.append(_manual_subtract_2d(arr[:, :, i], manual_value))
            return np.stack(chs, axis=2)
        return arr

    if arr.ndim == 2:
        return _dark_subtract_2d(arr, dark_pct) if mode == "dark_subtract" else _apply_rolling_ball(arr, radius)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        chs = []
        for i in range(3):
            ch = arr[:, :, i]
            chs.append(_dark_subtract_2d(ch, dark_pct) if mode == "dark_subtract" else _apply_rolling_ball(ch, radius))
        return np.stack(chs, axis=2)
    return arr

def normalize_array(arr: np.ndarray, enabled: bool, method: str) -> np.ndarray:
    if not enabled:
        return arr
    if arr.ndim == 2:
        return _normalize_channel(arr, method)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.stack([_normalize_channel(arr[:, :, i], method) for i in range(3)], axis=2)
    return arr

def preprocess_for_processing(
    img: Image.Image,
    *,
    use_native_scale: bool,
    bg_enable: bool,
    bg_radius: int,
    bg_mode: str = "rolling",
    bg_dark_pct: float = 5.0,
    norm_enable: bool = False,
    norm_method: str = "z-score",
    manual_background: float | None = None,
) -> np.ndarray:
    arr = pil_to_numpy_native(img) if use_native_scale else pil_to_numpy(img)
    arr = background_correction(
        arr, int(bg_radius), bool(bg_enable), mode=bg_mode, dark_pct=float(bg_dark_pct), manual_value=manual_background
    )
    arr = normalize_array(arr, bool(norm_enable), norm_method)
    return arr.astype(np.float32, copy=False)

def compute_dark_background(img: Image.Image, chan, pct: float, use_native_scale: bool = True) -> float:
    arr = pil_to_numpy_native(img) if use_native_scale else pil_to_numpy(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        ch = extract_single_channel(arr, chan)
    else:
        ch = arr.astype(np.float32, copy=False)
    p = float(np.clip(pct, 0.0, 50.0))
    try:
        thr = float(np.percentile(ch, p))
        dark = ch[ch <= thr]
        if dark.size == 0:
            return 0.0
        return float(np.mean(dark))
    except Exception:
        return 0.0

def extract_single_channel(img: np.ndarray, chan) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    key = chan
    if isinstance(chan, str):
        key = chan.strip().lower()
    elif isinstance(chan, (int, np.integer)):
        key = {0: "gray", 1: "r", 2: "g", 3: "b"}.get(int(chan), None)
    if key is None:
        raise ValueError("Invalid channel selection")
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if key in ("gray", "grey"):
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    if key == "r":
        return r.astype(np.float32)
    if key == "g":
        return g.astype(np.float32)
    if key == "b":
        return b.astype(np.float32)
    raise ValueError("Invalid channel selection")

def global_threshold_mask(img_gray: np.ndarray, nonsat_mask: np.ndarray, mode: str, pct: float, min_obj_size: int) -> Tuple[np.ndarray, float]:
    valid = img_gray[nonsat_mask]
    if valid.size == 0:
        raise ValueError("All pixels are saturated; lower the saturation limit")
    if mode == "global_otsu":
        th = float(filters.threshold_otsu(valid))
    elif mode == "global_percentile":
        th = float(np.percentile(valid, float(np.clip(pct, 0.0, 100.0))))
    else:
        raise ValueError("global_threshold_mask called with invalid mode")
    mask = (img_gray >= th) & nonsat_mask
    if min_obj_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_obj_size))
        mask = morphology.binary_opening(mask, morphology.disk(1))
    return mask, th

def per_cell_threshold(cell_pixels: np.ndarray, mode: str, pct: float) -> float:
    if cell_pixels.size == 0:
        return np.nan
    if mode == "per_cell_otsu":
        return float(filters.threshold_otsu(cell_pixels))
    if mode == "per_cell_percentile":
        return float(np.percentile(cell_pixels, float(np.clip(pct, 0.0, 100.0))))
    raise ValueError("per_cell_threshold called with invalid mode")

def cleanup_mask(mask: np.ndarray, min_obj_size: int) -> np.ndarray:
    if min_obj_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_obj_size))
        mask = morphology.binary_opening(mask, morphology.disk(1))
    return mask

def run_segmentation(
    target_img: Image.Image,
    reference_img: Image.Image,
    seg_source: str,
    seg_channel: int,
    diameter: float,
    flow_threshold: float,
    cellprob_threshold: float,
    use_gpu: bool,
    drop_edge_cells: bool = True,
    inside_fraction_min: float = 0.9,
    edge_margin_pct: float = 0.0,
):
    """Run Cellpose-SAM segmentation and optionally drop edge-touching cells.

    The previous logic removed any cell whose bounding box touched the image border.
    This was sometimes too strict: cells barely touching the border (e.g. one or two
    pixels) were discarded despite >90% of the cell area lying inside the field.

    New rule when drop_edge_cells=True:
        For each labeled cell we compute:
            near_border_pixels = count of cell pixels that lie within edge_margin_pct (%) from the image border
            area_pixels   = total cell pixels
            inside_fraction = 1 - near_border_pixels / area_pixels
        We only DROP the cell if inside_fraction < inside_fraction_min (default 0.9).

    This keeps cells that are mostly interior, while removing those largely truncated
    at the border. If desired, inside_fraction_min can be adjusted (e.g. 0.95 for stricter
    filtering or 0.8 for more permissive).
    """
    tgt = pil_to_numpy(target_img)
    ref = pil_to_numpy(reference_img)
    if tgt.shape[:2] != ref.shape[:2]:
        raise ValueError(f"Image size mismatch: target {tgt.shape[:2]} vs reference {ref.shape[:2]}")
    seg_arr = tgt if seg_source == "target" else ref
    seg_gray = extract_single_channel(seg_arr, seg_channel)
    model = get_model(use_gpu)
    result = model.eval(
        seg_gray,
        diameter=None if diameter <= 0 else diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0],
        normalize=True,
        invert=False,
        compute_masks=True,
        progress=None,
    )
    if isinstance(result, tuple) and len(result) == 4:
        masks, flows, styles, diams = result
    elif isinstance(result, tuple) and len(result) == 3:
        masks, flows, styles = result
    else:
        raise ValueError("Unexpected return values from model.eval")
    # Optionally drop cells that touch image borders and relabel to compact integers
    if drop_edge_cells and isinstance(masks, np.ndarray) and masks.ndim == 2 and masks.size > 0:
        H, W = masks.shape
        # Convert percentage margin to pixel margins per axis
        m_r = int(max(0, np.floor((float(edge_margin_pct) / 100.0) * H)))
        m_c = int(max(0, np.floor((float(edge_margin_pct) / 100.0) * W)))
        n_labels = int(masks.max())
        keep = np.ones(n_labels + 1, dtype=bool)
        keep[0] = False
        try:
            props = measure.regionprops(masks)
            for p in props:
                lab = int(p.label)
                coords = p.coords  # (N, 2) array of (row, col)
                if coords.size == 0:
                    keep[lab] = False
                    continue
                rows = coords[:, 0]; cols = coords[:, 1]
                # Count pixels within the specified margin from the border
                if (m_r <= 0) and (m_c <= 0):
                    near_mask = (rows == 0) | (rows == H - 1) | (cols == 0) | (cols == W - 1)
                else:
                    near_mask = (rows < m_r) | (rows > H - 1 - m_r) | (cols < m_c) | (cols > W - 1 - m_c)
                border_pixels = int(np.count_nonzero(near_mask))
                area_pixels = int(coords.shape[0])
                inside_fraction = 1.0 - (border_pixels / area_pixels) if area_pixels > 0 else 0.0
                if inside_fraction < float(inside_fraction_min):
                    keep[lab] = False
        except Exception:
            # Fallback (rare): simple edge-touching removal using dilation of border
            edge_mask = np.zeros_like(masks, dtype=bool)
            edge_mask[0, :] = True; edge_mask[-1, :] = True; edge_mask[:, 0] = True; edge_mask[:, -1] = True
            labs_on_edge = np.unique(masks[edge_mask])
            for lab in labs_on_edge:
                if lab > 0:
                    keep[int(lab)] = False
        # Apply filter & relabel compactly WITHOUT merging adjacent kept labels.
        # Note: converting to binary and calling measure.label merges touching labels.
        # To preserve original separations, remap kept labels to a compact range directly.
        kept_labels = np.flatnonzero(keep)
        kept_labels = kept_labels[kept_labels > 0]
        if kept_labels.size == 0:
            masks = np.zeros_like(masks, dtype=np.int32)
        else:
            new_masks = np.zeros_like(masks, dtype=np.int32)
            # Map each kept label to a new consecutive id, preserving boundaries
            for new_id, lab in enumerate(kept_labels, start=1):
                new_masks[masks == int(lab)] = int(new_id)
            masks = new_masks

    overlay = colorize_overlay(seg_gray, masks, None)
    overlay = annotate_ids(overlay, masks)
    mask_viz = vivid_label_image(masks)
    seg_tiff = save_label_tiff(masks, "seg_labels")
    return overlay, seg_tiff, mask_viz, masks

def apply_mask(
    img: Image.Image,
    masks: np.ndarray,
    measure_channel: int,
    sat_limit: float,
    mask_mode: str,
    pct: float,
    min_obj_size: int,
    roi_labels: Optional[np.ndarray] = None,
    mask_name: str = "mask",
):
    if masks is None:
        raise ValueError("Segmentation masks not provided. Run segmentation first.")
    gray = extract_single_channel(pil_to_numpy(img), measure_channel)
    orig01 = pil_to_numpy(img)
    gray_sat = extract_single_channel(orig01, measure_channel)
    sat_thr01 = _to_float_saturation_limit(img, float(sat_limit))
    nonsat = gray_sat < sat_thr01
    mask_total = np.zeros_like(masks, dtype=bool)
    global_mask = None
    if mask_mode in ("global_otsu", "global_percentile"):
        global_mask, _ = global_threshold_mask(gray, nonsat, mask_mode, float(pct), int(min_obj_size))
    labels = np.unique(masks); labels = labels[labels > 0]
    for lab in labels:
        region = (roi_labels == lab) if roi_labels is not None else (masks == lab)
        if mask_mode == "none":
            mask_cell = region & nonsat
        elif mask_mode in ("global_otsu", "global_percentile"):
            mask_cell = cleanup_mask(global_mask & region, int(min_obj_size))
        elif mask_mode in ("per_cell_otsu", "per_cell_percentile"):
            pool = gray[region & nonsat]
            th = per_cell_threshold(pool, mask_mode, float(pct))
            base = (gray >= th) & nonsat & region if not np.isnan(th) else np.zeros_like(region, dtype=bool)
            mask_cell = cleanup_mask(base, int(min_obj_size))
        else:
            raise ValueError("Invalid mask_mode")
        mask_total |= mask_cell
    overlay = colorize_overlay(gray, masks, mask_total)
    overlay = annotate_ids(overlay, masks)
    tiff_path = save_bool_mask_tiff(mask_total, mask_name)
    return overlay, tiff_path, mask_total

def integrate_and_quantify(
    target_img: Image.Image,
    reference_img: Image.Image,
    masks: np.ndarray,
    tgt_mask: np.ndarray,
    ref_mask: np.ndarray,
    tgt_chan: int,
    ref_chan: int,
    pixel_width_um: float,
    pixel_height_um: float,
    pp_bg_enable: bool,
    pp_bg_radius: int,
    pp_norm_enable: bool,
    pp_norm_method: str,
    *,
    bg_mode: str = "rolling",
    bg_dark_pct: float = 5.0,
    manual_tar_bg: float | None = None,
    manual_ref_bg: float | None = None,
    roi_mask: np.ndarray | None = None,
    roi_labels: np.ndarray | None = None,
    ratio_ref_epsilon: float = 0.0,
):
    if masks is None or tgt_mask is None or ref_mask is None:
        raise ValueError("Run previous steps first.")
    # Visualization arrays (0-1) for overlays/ratio
    tgt_vis = preprocess_for_processing(
        target_img, use_native_scale=False,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_tar_bg,
    )
    ref_vis = preprocess_for_processing(
        reference_img, use_native_scale=False,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_ref_bg,
    )
    if tgt_vis.shape[:2] != ref_vis.shape[:2]:
        raise ValueError("Image size mismatch between target and reference")
    # Native arrays (original scale) for measurements
    tgt_nat = preprocess_for_processing(
        target_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_tar_bg,
    )
    ref_nat = preprocess_for_processing(
        reference_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_ref_bg,
    )
    tgt_gray_vis = extract_single_channel(tgt_vis, tgt_chan)
    ref_gray_vis = extract_single_channel(ref_vis, ref_chan)
    tgt_gray_nat = extract_single_channel(tgt_nat, tgt_chan)
    ref_gray_nat = extract_single_channel(ref_nat, ref_chan)
    # Base AND mask (inside cells only)
    and_mask = tgt_mask & ref_mask
    # Build a selection mask for visualization and measurement.
    # If roi_labels provided (from radial), include outside-of-cell ring per cell label.
    selection_mask = None
    if roi_mask is not None and roi_labels is not None:
        selection_mask = np.zeros_like(and_mask, dtype=bool)
        labels = np.unique(masks); labels = labels[labels > 0]
        for lab in labels:
            cell = (masks == lab)
            # inside: respect AND mask gating within cell and ROI
            inside = and_mask & cell & roi_mask & (roi_labels == lab)
            # outside: ROI-labeled region assigned to this cell and outside the cell
            outside = (roi_mask & (roi_labels == lab) & (~cell))
            selection_mask |= (inside | outside)
    elif roi_mask is not None:
        # Without labels, just intersect ROI with AND mask
        selection_mask = and_mask & roi_mask
    else:
        selection_mask = and_mask
    labels = np.unique(masks); labels = labels[labels > 0]
    try:
        px_area_um2 = float(pixel_width_um) * float(pixel_height_um)
    except Exception:
        px_area_um2 = 1.0
    rows = []
    for lab in labels:
        cell = masks == lab
        if roi_mask is not None and roi_labels is not None:
            # Per-label selection includes ROI outside the cell for this label
            idx_inside = and_mask & cell & roi_mask & (roi_labels == lab)
            idx_out = roi_mask & (roi_labels == lab) & (~cell)
            idx = idx_inside | idx_out
        elif roi_mask is not None:
            idx = (and_mask & roi_mask) & cell
        else:
            idx = and_mask & cell
        area_cell_px = int(cell.sum())
        area_and_px = int(idx.sum())
        area_cell_um2 = float(area_cell_px) * px_area_um2
        area_and_um2 = float(area_and_px) * px_area_um2
        if area_and_px == 0:
            mean_t_mem = std_t_mem = sum_t_mem = mean_r_mem = std_r_mem = sum_r_mem = ratio_mean = ratio_sum = ratio_std = np.nan
            ratio_of_means_all = np.nan
        else:
            mean_t_mem = float(np.mean(tgt_gray_nat[idx]))
            std_t_mem = float(np.std(tgt_gray_nat[idx]))
            sum_t_mem = float(np.sum(tgt_gray_nat[idx]))
            mean_r_mem = float(np.mean(ref_gray_nat[idx]))
            std_r_mem = float(np.std(ref_gray_nat[idx]))
            sum_r_mem = float(np.sum(ref_gray_nat[idx]))
            # Smoothed ratio over the SAME set: (T+eps)/(R+eps)
            r_sub = ref_gray_nat[idx].astype(np.float64)
            t_sub = tgt_gray_nat[idx].astype(np.float64)
            eps = max(1e-12, float(ratio_ref_epsilon))
            ratio_vals = (t_sub + eps) / (r_sub + eps)
            ratio_mean = float(np.mean(ratio_vals)) if ratio_vals.size > 0 else np.nan
            ratio_std = float(np.std(ratio_vals)) if ratio_vals.size > 1 else np.nan
            ratio_sum = float(np.sum(ratio_vals)) if ratio_vals.size > 0 else np.nan
            # Ratio of means (smoothed) on ALL selected pixels
            ratio_of_means_all = ((mean_t_mem + eps) / (mean_r_mem + eps)) if np.isfinite(mean_t_mem) and np.isfinite(mean_r_mem) else np.nan
            # no per-valid-set ratio_of_means column anymore
        mean_t_whole = float(np.mean(tgt_gray_nat[cell])) if area_cell_px > 0 else np.nan
        std_t_whole = float(np.std(tgt_gray_nat[cell])) if area_cell_px > 0 else np.nan
        sum_t_whole = float(np.sum(tgt_gray_nat[cell])) if area_cell_px > 0 else np.nan
        mean_r_whole = float(np.mean(ref_gray_nat[cell])) if area_cell_px > 0 else np.nan
        std_r_whole = float(np.std(ref_gray_nat[cell])) if area_cell_px > 0 else np.nan
        sum_r_whole = float(np.sum(ref_gray_nat[cell])) if area_cell_px > 0 else np.nan
        rows.append({
            "label": int(lab),
            "area_cell_px": area_cell_px,
            "area_cell_um2": area_cell_um2,
            "area_and_px": area_and_px,
            "area_and_um2": area_and_um2,
            "sum_target_on_mask": sum_t_mem,
            "mean_target_on_mask": mean_t_mem,
            "std_target_on_mask": std_t_mem,
            "sum_reference_on_mask": sum_r_mem,
            "mean_reference_on_mask": mean_r_mem,
            "std_reference_on_mask": std_r_mem,
            "sum_ratio_T_over_R": ratio_sum,
            "mean_ratio_T_over_R": ratio_mean,
            "std_ratio_T_over_R": ratio_std,
            "ratio_of_means_on_mask": ratio_of_means_all,
            "sum_target_whole": sum_t_whole,
            "mean_target_whole": mean_t_whole,
            "std_target_whole": std_t_whole,
            "sum_reference_whole": sum_r_whole,
            "mean_reference_whole": mean_r_whole,
            "std_reference_whole": std_r_whole,
        })
    df = pd.DataFrame(rows).sort_values("label")
    if "label" in df.columns:
        df = df[["label"] + [c for c in df.columns if c != "label"]]
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_csv.name, index=False)
    overlay_tar = colorize_overlay(tgt_gray_vis, masks, selection_mask)
    overlay_tar = annotate_ids(overlay_tar, masks)
    overlay_ref = colorize_overlay(ref_gray_vis, masks, selection_mask)
    overlay_ref = annotate_ids(overlay_ref, masks)
    and_tiff = save_bool_mask_tiff(selection_mask, "and_mask")
    tgt_on_and = Image.fromarray((np.clip(np.where(selection_mask, tgt_gray_vis, 0.0), 0, 1) * 255).astype(np.uint8))
    ref_on_and = Image.fromarray((np.clip(np.where(selection_mask, ref_gray_vis, 0.0), 0, 1) * 255).astype(np.uint8))
    eps = max(1e-12, float(ratio_ref_epsilon))
    ratio_img = (tgt_gray_vis + eps) / (ref_gray_vis + eps)
    ratio_masked = np.where(selection_mask, ratio_img, np.nan)
    finite_vals = ratio_masked[np.isfinite(ratio_masked)]
    if finite_vals.size > 0:
        vmin = np.percentile(finite_vals, 1.0)
        vmax = np.percentile(finite_vals, 99.0)
        vmax = vmin + 1e-6 if vmax <= vmin else vmax
        ratio_norm = (ratio_masked - vmin) / (vmax - vmin)
    else:
        ratio_norm = np.zeros_like(ratio_masked, dtype=np.float32)
    ratio_overlay = Image.fromarray((np.nan_to_num(ratio_norm, nan=0.0) * 255).astype(np.uint8))
    return overlay_tar, overlay_ref, and_tiff, df, tmp_csv.name, tgt_on_and, ref_on_and, ratio_overlay

def _apply_rolling_ball(channel: np.ndarray, radius: int) -> np.ndarray:
    radius = int(max(1, radius))
    try:
        bg = restoration.rolling_ball(channel, radius=radius)
        out = channel - bg
        return np.clip(out, 0, None)
    except Exception:
        return channel

def _normalize_channel(ch: np.ndarray, method: str) -> np.ndarray:
    method = (method or "z-score").strip().lower()
    eps = 1e-12
    if method == "z-score":
        mu, sigma = float(np.nanmean(ch)), float(np.nanstd(ch))
        return (ch - mu) / (sigma + eps)
    if method in ("min-max", "minmax"):
        vmin, vmax = float(np.nanmin(ch)), float(np.nanmax(ch))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return ch * 0.0
        return (ch - vmin) / (vmax - vmin)
    if method.startswith("percentile"):
        vmin = float(np.nanpercentile(ch, 1.0))
        vmax = float(np.nanpercentile(ch, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return ch * 0.0
        return np.clip((ch - vmin) / (vmax - vmin), 0.0, 1.0)
    if method in ("robust z-score", "robust_z", "robust"):
        med = float(np.nanmedian(ch))
        mad = float(np.nanmedian(np.abs(ch - med)))
        sigma = 1.4826 * mad
        return (ch - med) / (sigma + eps)
    # default
    mu, sigma = float(np.nanmean(ch)), float(np.nanstd(ch))
    return (ch - mu) / (sigma + eps)

def _to_float_saturation_limit(img: Image.Image, sat_limit_abs_8bit: float) -> float:
    raw = np.array(img)
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    ref_max = 255.0
    if np.issubdtype(raw.dtype, np.integer):
        native_max = float(np.iinfo(raw.dtype).max)
    elif np.issubdtype(raw.dtype, np.floating):
        vmax = float(np.nanmax(raw)) if raw.size > 0 else 1.0
        native_max = 1.0 if vmax <= 1.0 else vmax
    else:
        native_max = 1.0
    if native_max <= ref_max:
        sat_native = float(np.clip(sat_limit_abs_8bit, 0.0, native_max))
    else:
        sat_native = float(np.clip(sat_limit_abs_8bit * (native_max / ref_max), 0.0, native_max))
    thr01 = sat_native / (native_max if native_max > 0 else 1.0)
    return float(np.clip(thr01, 0.0, 1.0))

