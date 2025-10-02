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

# -----------------------
# Display/label settings (adjustable via UI)
# -----------------------
LABEL_SCALE: float = 1.8


def _load_font(point_size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
        "NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/ARIALUNI.TTF",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, point_size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("arial.ttf", point_size)
    except Exception:
        return ImageFont.load_default()


# -----------------------
# Model lazy loader
# -----------------------
_MODEL: Optional[models.CellposeModel] = None


def get_model(use_gpu: bool = False) -> models.CellposeModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    return _MODEL


# -----------------------
# Image helpers
# -----------------------

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


# -----------------------
# Preprocess helpers (applied only at Integrate step)
# -----------------------

def _apply_rolling_ball(channel: np.ndarray, radius: int) -> np.ndarray:
    radius = int(max(1, radius))
    try:
        bg = restoration.rolling_ball(channel, radius=radius)
        out = channel - bg
        return np.clip(out, 0, None)
    except Exception:
        return channel


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


# -----------------------
# TIFF helpers
# -----------------------

def arr01_to_pil_for_preview(arr: np.ndarray) -> Image.Image:
    a = arr
    eps = 1e-6
    try:
        amin = float(np.nanmin(a)); amax = float(np.nanmax(a))
    except Exception:
        amin = 0.0; amax = 0.0
    if amin >= 0.0 and amax <= 1.0:
        clipped = np.clip(a, 0.0, 1.0)
        if clipped.ndim == 2:
            return Image.fromarray((clipped * 255).astype(np.uint8))
        if clipped.ndim == 3 and clipped.shape[2] >= 3:
            out = (np.clip(clipped[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
            return Image.fromarray(out)
    if a.ndim == 2:
        vmin = float(np.nanpercentile(a, 1.0)); vmax = float(np.nanpercentile(a, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= eps:
            vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
        denom = (vmax - vmin) if (vmax - vmin) > eps else 1.0
        a_scaled = np.clip((a - vmin) / denom, 0.0, 1.0)
        return Image.fromarray((a_scaled * 255).astype(np.uint8))
    if a.ndim == 3 and a.shape[2] >= 3:
        chs = []
        for i in range(3):
            ch = a[:, :, i]
            vmin = float(np.nanpercentile(ch, 1.0)); vmax = float(np.nanpercentile(ch, 99.0))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= eps:
                vmin, vmax = float(np.nanmin(ch)), float(np.nanmax(ch))
            denom = (vmax - vmin) if (vmax - vmin) > eps else 1.0
            chs.append(np.clip((ch - vmin) / denom, 0.0, 1.0))
        out = (np.stack(chs, axis=2) * 255).astype(np.uint8)
        return Image.fromarray(out)
    a_clipped = np.clip(a, 0.0, 1.0)
    return Image.fromarray((a_clipped * 255).astype(np.uint8))


def save_bool_mask_tiff(mask: np.ndarray, stem: str) -> str:
    arr = (mask.astype(np.uint8) * 255)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.tif")
    Image.fromarray(arr).save(tmp.name)
    return tmp.name


def save_label_tiff(labels: np.ndarray, stem: str) -> str:
    maxv = int(labels.max()) if labels.size > 0 else 0
    if maxv <= 65535:
        out = labels.astype(np.uint16, copy=False)
    else:
        out = labels.astype(np.float32, copy=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.tif")
    Image.fromarray(out).save(tmp.name)
    return tmp.name


# -----------------------
# Saturation helper
# -----------------------

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


# -----------------------
# Channel extraction
# -----------------------

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


# -----------------------
# Visualization helpers
# -----------------------

def colorize_overlay(image_gray: np.ndarray, masks: np.ndarray, vis_mask: Optional[np.ndarray]) -> Image.Image:
    edges = np.zeros_like(masks, dtype=bool)
    if masks.max() > 0:
        for lab in np.unique(masks):
            if lab == 0:
                continue
            obj = masks == lab
            er = ndi.binary_erosion(obj, iterations=1, border_value=0)
            edges |= obj ^ er
    base = (np.clip(image_gray, 0, 1) * 255).astype(np.uint8)
    overlay = np.stack([base, base, base], axis=2)
    overlay[edges] = [255, 0, 0]
    if vis_mask is not None:
        er = ndi.binary_erosion(vis_mask, iterations=1)
        vis_edges = vis_mask ^ er
        overlay[vis_edges] = [0, 255, 0]
    return Image.fromarray(overlay)


def vivid_label_image(masks: np.ndarray) -> Image.Image:
    lbl_rgb = color.label2rgb(masks, bg_label=0, bg_color=(0, 0, 0), alpha=1.0)
    img = (np.clip(lbl_rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img)


def annotate_ids(img: Image.Image, masks: np.ndarray) -> Image.Image:
    try:
        if float(LABEL_SCALE) <= 0.0:
            return img
    except Exception:
        pass
    draw = ImageDraw.Draw(img)
    w, h = img.size
    base = max(14, int(min(w, h) * 0.035))
    fsize = max(12, int(base * float(LABEL_SCALE)))
    font = _load_font(fsize)
    props = measure.regionprops(masks)
    for p in props:
        lab = p.label
        cy, cx = p.centroid
        x, y = int(cx), int(cy)
        text = str(lab)
        outline_color = (0, 0, 0)
        off = max(2, fsize // 12)
        for dx, dy in [(-off,0),(off,0),(0,-off),(0,off),(-off,-off),(off,off),(-off,off),(off,-off)]:
            draw.text((x+dx, y+dy), text, fill=outline_color, font=font, anchor="mm")
        draw.text((x, y), text, fill=(255, 255, 0), font=font, anchor="mm")
    return img


# -----------------------
# Threshold utilities
# -----------------------

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


# -----------------------
# Step 1: Cellpose segmentation (raw images)
# -----------------------

def run_segmentation(
    target_img: Image.Image,
    reference_img: Image.Image,
    seg_source: str,
    seg_channel: int,
    diameter: float,
    flow_threshold: float,
    cellprob_threshold: float,
    use_gpu: bool,
):
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
    overlay = colorize_overlay(seg_gray, masks, None)
    overlay = annotate_ids(overlay, masks)
    mask_viz = vivid_label_image(masks)
    seg_tiff = save_label_tiff(masks, "seg_labels")
    return overlay, seg_tiff, mask_viz, masks


# -----------------------
# Step 2: Radial mask (optional)
# -----------------------

def radial_mask(
    masks: np.ndarray,
    inner_pct: float,
    outer_pct: float,
    min_obj_size: int,
):
    if masks is None:
        raise ValueError("Segmentation masks not provided. Run segmentation first.")
    labels = np.unique(masks); labels = labels[labels > 0]
    if labels.size == 0:
        raise ValueError("No cells found in masks.")
    H, W = masks.shape
    bg = masks == 0
    props_map = {p.label: p for p in measure.regionprops(masks)}
    radial_total = np.zeros_like(masks, dtype=bool)
    radial_labels = np.zeros_like(masks, dtype=masks.dtype)
    dmax_table = {}
    for lab in labels:
        cell = masks == lab
        p = props_map.get(int(lab))
        if p is None or p.area <= 0:
            continue
        di = ndi.distance_transform_edt(cell)
        dmax = float(di.max())
        if dmax <= 0:
            radial_total |= cell
            continue
        dmax_table[int(lab)] = dmax
        t = 1.0 - (di / dmax)
        rin_t = max(0.0, float(inner_pct)) / 100.0
        rout_t = max(0.0, float(outer_pct)) / 100.0
        if rin_t > rout_t:
            rin_t, rout_t = rout_t, rin_t
        inside_band = (t >= rin_t) & (t <= min(rout_t, 1.0)) & cell
        radial_cell = cleanup_mask(inside_band.copy(), int(min_obj_size))
        radial_total |= radial_cell
        radial_labels[radial_cell] = int(lab)
    extra_frac = max(0.0, float(outer_pct) / 100.0 - 1.0)
    if extra_frac > 0.0 and len(dmax_table) > 0:
        fg = masks > 0
        dist_bg, idx = ndi.distance_transform_edt(~fg, return_indices=True)
        nearest_label = masks[idx[0], idx[1]]
        max_lab = int(masks.max())
        extra_table = np.zeros(max_lab + 1, dtype=np.float32)
        for lab, dmax in dmax_table.items():
            extra_table[int(lab)] = extra_frac * float(dmax)
        outside_bool = (~fg) & (nearest_label > 0) & (dist_bg > 0) & (dist_bg <= extra_table[nearest_label])
        add_mask = outside_bool & bg
        radial_total |= add_mask
        to_set = (radial_labels == 0) & add_mask
        radial_labels[to_set] = nearest_label[to_set].astype(radial_labels.dtype)
    overlay = colorize_overlay((masks > 0).astype(np.float32), masks, radial_total)
    overlay = annotate_ids(overlay, masks)
    rad_bool_tiff = save_bool_mask_tiff(radial_total, "radial_mask")
    rad_lbl_tiff = save_label_tiff(radial_labels, "radial_labels")
    return overlay, radial_total, radial_labels, rad_bool_tiff, rad_lbl_tiff


# -----------------------
# Step 3/4: Apply mask for target or reference (raw)
# -----------------------

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


# -----------------------
# Step 5: Integrate masks and quantify (preprocess here)
# -----------------------

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
        else:
            mean_t_mem = float(np.mean(tgt_gray_nat[idx]))
            std_t_mem = float(np.std(tgt_gray_nat[idx]))
            sum_t_mem = float(np.sum(tgt_gray_nat[idx]))
            mean_r_mem = float(np.mean(ref_gray_nat[idx]))
            std_r_mem = float(np.std(ref_gray_nat[idx]))
            sum_r_mem = float(np.sum(ref_gray_nat[idx]))
            if np.all(ref_gray_nat[idx] > 0):
                ratio_vals = tgt_gray_nat[idx] / ref_gray_nat[idx]
                ratio_mean = float(np.mean(ratio_vals))
                ratio_std = float(np.std(ratio_vals))
                ratio_sum = float(np.sum(ratio_vals))
            else:
                ratio_mean = ratio_std = ratio_sum = np.nan
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
    valid = (ref_gray_vis > 0)
    ratio_img = np.full_like(tgt_gray_vis, np.nan, dtype=np.float32)
    ratio_img[valid] = tgt_gray_vis[valid] / ref_gray_vis[valid]
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


# -----------------------
# UI (Single tab temporarily disabled)
# -----------------------

def build_ui():
    with gr.Blocks(title="DualCellQuant") as demo:
        gr.Markdown(
            """
            # 🔬 **DualCellQuant**
            *Segment, filter, and compare cells across two fluorescence channels*
            1. **Run Cellpose-SAM** to obtain segmentation masks.
            2. **Build Radial mask** (optional).
            3. **Apply Target/Reference masks**.
            4. **Integrate** (Preprocess applied only here) and view results.
            """
        )
        with gr.Tabs():
            with gr.TabItem("Dual images"):
                masks_state = gr.State()
                radial_mask_state = gr.State()
                radial_label_state = gr.State()
                tgt_mask_state = gr.State()
                ref_mask_state = gr.State()
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            tgt = gr.Image(type="pil", label="Target image", image_mode="RGB", width=600)
                            ref = gr.Image(type="pil", label="Reference image", image_mode="RGB", width=600)
                        
                        with gr.Accordion("Settings", open=False):
                            reset_settings = gr.Button("Reset Settings",scale=1)
                            label_scale = gr.Slider(0.0, 5.0, value=float(LABEL_SCALE), step=0.1, label="Label size scale (0=hidden)")
                        with gr.Accordion("Segmentation params", open=False):
                            seg_source = gr.Radio(["target","reference"], value="target", label="Segment on")
                            seg_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Segmentation channel")
                            diameter = gr.Slider(0, 200, value=0, step=1, label="Diameter (px, 0=auto)")
                            flow_th = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                            cellprob_th = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")
                            use_gpu = gr.Checkbox(value=True, label="Use GPU if available")
                        run_seg_btn = gr.Button("1. Run Cellpose")
                        with gr.Row():
                            seg_overlay = gr.Image(type="pil", label="Segmentation overlay", width=600)
                            mask_img = gr.Image(type="pil", label="Segmentation label image", width=600)
                        seg_tiff_file = gr.File(label="Download masks (label TIFF)")
                        # Radial mask controls are moved to the bottom (after Integrate)
                        with gr.Accordion("Apply mask", open=False):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**Target mask settings**")
                                    tgt_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Target channel")
                                    tgt_mask_mode = gr.Dropdown(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                                    tgt_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                                    tgt_sat_limit = gr.Slider(0, 255, value=254, step=1, label="Saturation limit (abs, 8-bit scale)")
                                    tgt_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                                with gr.Column():
                                    gr.Markdown("**Reference mask settings**")
                                    ref_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Reference channel")
                                    ref_mask_mode = gr.Dropdown(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                                    ref_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                                    ref_sat_limit = gr.Slider(0, 255, value=254, step=1, label="Saturation limit (abs, 8-bit scale)")
                                    ref_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                        run_tgt_btn = gr.Button("2. Apply Target & Reference masks")
                        with gr.Row():
                            with gr.Column():
                                tgt_overlay = gr.Image(type="pil", label="Target mask overlay", width=600)
                                tgt_tiff = gr.File(label="Download target mask (TIFF)")

                            with gr.Column():
                                ref_overlay = gr.Image(type="pil", label="Reference mask overlay", width=600)
                                ref_tiff = gr.File(label="Download reference mask (TIFF)")
                        with gr.Accordion("Integrate & Quantify", open=False):
                            with gr.Column():
                                pp_bg_enable = gr.Checkbox(value=False, label="Background correction")
                                pp_bg_mode = gr.Dropdown(["rolling","dark_subtract","manual"], value="dark_subtract", label="BG method")
                                pp_bg_radius = gr.Slider(1, 300, value=50, step=1, label="Rolling ball radius (px)")
                                pp_dark_pct = gr.Slider(0.0, 50.0, value=5.0, step=0.5, label="Dark percentile (%)")
                                
                                with gr.Row():
                                    bak_tar = gr.Number(value=1.0, label="Target background", scale=1)
                                    bak_ref = gr.Number(value=1.0, label="Reference background", scale=1)
                            with gr.Column():
                                pp_norm_enable = gr.Checkbox(value=False, label="Normalization")
                                pp_norm_method = gr.Dropdown([
                                    "z-score",
                                    "robust z-score",
                                    "min-max",
                                    "percentile [1,99]",
                                ], value="min-max", label="Normalization method")
                            with gr.Column():
                                with gr.Row():
                                    px_w = gr.Number(value=1.0, label="Pixel width (µm)", scale=1)
                                    px_h = gr.Number(value=1.0, label="Pixel height (µm)", scale=1)
                        integrate_btn = gr.Button("4. Integrate & Quantify")
                        with gr.Row():
                            integrate_tar_overlay = gr.Image(type="pil", label="Integrate Target overlay (AND mask)", width=600)
                            integrate_ref_overlay = gr.Image(type="pil", label="Integrate Reference overlay (AND mask)", width=600)

                        mask_tiff = gr.File(label="Download AND mask (TIFF)")
                        table = gr.Dataframe(label="Per-cell intensities & ratios", interactive=False, pinned_columns=1)
                        csv_file = gr.File(label="Download CSV")
                        with gr.Row():
                            tgt_on_and_img = gr.Image(type="pil", label="Target on AND mask", width=600)
                            ref_on_and_img = gr.Image(type="pil", label="Reference on AND mask", width=600)
                            ratio_img = gr.Image(type="pil", label="Ratio (Target/Reference) on AND mask", width=600)
                        # Radial mask section moved here (after integration)
                        with gr.Accordion("Radial mask (optional, after integration)", open=False):
                            rad_in = gr.Slider(0.0, 150.0, value=0.0, step=1.0, label="Radial inner % (0=中心)")
                            rad_out = gr.Slider(0.0, 150.0, value=100.0, step=1.0, label="Radial outer % (100=境界)")
                            rad_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                        run_rad_btn = gr.Button("5. Build Radial mask")
                        # rad_overlay = gr.Image(type="pil", label="Radial mask overlay", width=600)
                        rad_overlay=gr.State()
                        with gr.Row():
                            radial_tar_overlay = gr.Image(type="pil", label="Integrate Target overlay (Radial AND mask)", width=600)
                            radial_ref_overlay = gr.Image(type="pil", label="Integrate Reference overlay (Radial AND mask)", width=600)
                        rad_tiff = gr.File(label="Download radial mask (TIFF)")
                        # rad_lbl_tiff = gr.File(label="Download radial labels (label TIFF)")
                        rad_lbl_tiff=gr.State()
                        radial_table = gr.Dataframe(label="Radial per-cell intensities & ratios", interactive=False, pinned_columns=1)
                        radial_csv = gr.File(label="Download radial CSV")

                        with gr.Row():
                            radial_tgt_on_and_img = gr.Image(type="pil", label="Target on Radial AND mask", width=600)
                            radial_ref_on_and_img = gr.Image(type="pil", label="Reference on Radial AND mask", width=600)
                            radial_ratio_img = gr.Image(type="pil", label="Ratio (Target/Reference) on Radial AND mask", width=600)
                        

                # Segmentation
                def _run_seg(tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu):
                    return run_segmentation(tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu)
                run_seg_btn.click(
                    fn=_run_seg,
                    inputs=[tgt, ref, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu],
                    outputs=[seg_overlay, seg_tiff_file, mask_img, masks_state],
                )
                # Radial mask (now at bottom)
                def _radial_and_quantify(tgt_img, ref_img, masks, rin, rout, mino, tmask, rmask, tchan, rchan, pw, ph, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r):
                    ov, rad_bool, rad_lbl, tiff_bool, tiff_lbl = radial_mask(masks, rin, rout, mino)
                    # choose manual backgrounds only when mode is manual
                    bgm = str(bg_mode)
                    mt = float(man_t) if (bg_en and bgm == "manual") else None
                    mr = float(man_r) if (bg_en and bgm == "manual") else None
                    q_tar_ov, q_ref_ov, q_and_tiff, q_df, q_csv, q_tgt_on, q_ref_on, q_ratio = integrate_and_quantify(
                        tgt_img, ref_img, masks, tmask, rmask, tchan, rchan, pw, ph,
                        bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                        bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                        manual_tar_bg=mt, manual_ref_bg=mr, roi_mask=rad_bool, roi_labels=rad_lbl,
                    )
                    return ov, rad_bool, rad_lbl, tiff_bool, tiff_lbl, q_df, q_csv, q_tar_ov, q_ref_ov, q_tgt_on, q_ref_on, q_ratio
                run_rad_btn.click(
                    fn=_radial_and_quantify,
                    inputs=[tgt, ref, masks_state, rad_in, rad_out, rad_min_obj, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan, px_w, px_h, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref],
                    outputs=[rad_overlay, radial_mask_state, radial_label_state, rad_tiff, rad_lbl_tiff, radial_table, radial_csv, radial_tar_overlay, radial_ref_overlay, radial_tgt_on_and_img, radial_ref_on_and_img, radial_ratio_img],
                )
                # Target/Reference masking (no ROI coupling)
                def _apply_mask_generic(img, m, ch, sat, mode, p, mino, name):
                    return apply_mask(img, m, ch, sat, mode, p, mino, None, name)
                # Combined: apply both masks in one click from the Target button
                def _apply_masks_both(tgt_img, ref_img, m, t_ch, t_sat, t_mode, t_p, t_mino, r_ch, r_sat, r_mode, r_p, r_mino):
                    t_ov, t_tiff_path, t_mask = apply_mask(tgt_img, m, t_ch, t_sat, t_mode, t_p, t_mino, None, "target_mask")
                    r_ov, r_tiff_path, r_mask = apply_mask(ref_img, m, r_ch, r_sat, r_mode, r_p, r_mino, None, "reference_mask")
                    return t_ov, t_tiff_path, t_mask, r_ov, r_tiff_path, r_mask
                run_tgt_btn.click(
                    fn=_apply_masks_both,
                    inputs=[tgt, ref, masks_state, tgt_chan, tgt_sat_limit, tgt_mask_mode, tgt_pct, tgt_min_obj, ref_chan, ref_sat_limit, ref_mask_mode, ref_pct, ref_min_obj],
                    outputs=[tgt_overlay, tgt_tiff, tgt_mask_state, ref_overlay, ref_tiff, ref_mask_state],
                )

                # Toggle percentile slider visibility based on mask mode
                def _toggle_pct_vis(mode: str):
                    m = (str(mode) if mode is not None else '').lower()
                    return gr.update(visible=(m in ("global_percentile", "per_cell_percentile")))
                tgt_mask_mode.change(
                    fn=_toggle_pct_vis,
                    inputs=[tgt_mask_mode],
                    outputs=[tgt_pct],
                )
                ref_mask_mode.change(
                    fn=_toggle_pct_vis,
                    inputs=[ref_mask_mode],
                    outputs=[ref_pct],
                )

                # Ensure initial visibility on app load after settings are restored
                # def _init_pct_vis(t_mode: str, r_mode: str):
                #     tm = (str(t_mode) if t_mode is not None else '').lower()
                #     rm = (str(r_mode) if r_mode is not None else '').lower()
                #     return (
                #         gr.update(visible=(tm in ("global_percentile", "per_cell_percentile"))),
                #         gr.update(visible=(rm in ("global_percentile", "per_cell_percentile"))),
                #     )
                # demo.load(
                #     fn=_init_pct_vis,
                #     inputs=[tgt_mask_mode, ref_mask_mode],
                #     outputs=[tgt_pct, ref_pct],
                # )

                def _pp_bg_mode_changed_int(mode: str):
                    m = (mode or "rolling").lower()
                    return (
                        gr.update(visible=(m == "rolling")),
                        gr.update(visible=(m == "dark_subtract")),
                        gr.update(visible=(m in ("manual", "dark_subtract"))),
                        gr.update(visible=(m in ("manual", "dark_subtract"))),
                    )
                pp_bg_mode.change(
                    fn=_pp_bg_mode_changed_int,
                    inputs=[pp_bg_mode],
                    outputs=[pp_bg_radius, pp_dark_pct, bak_tar, bak_ref],
                )
                def _integrate_callback(tgt_img, ref_img, ms, tmask, rmask, tchan, rchan, pw, ph, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r):
                    # Decide manual backgrounds and display values
                    bg_mode_s = str(bg_mode)
                    out_tar_bg = man_t
                    out_ref_bg = man_r
                    if str(bg_en).lower() in ("true", "1") and bg_mode_s == "dark_subtract":
                        try:
                            out_tar_bg = compute_dark_background(tgt_img, tchan, float(dark_pct), use_native_scale=True)
                        except Exception:
                            out_tar_bg = man_t
                        try:
                            out_ref_bg = compute_dark_background(ref_img, rchan, float(dark_pct), use_native_scale=True)
                        except Exception:
                            out_ref_bg = man_r
                    # Prepare manual values to pass
                    man_t = float(out_tar_bg) if (bg_en and bg_mode_s == "manual") else None
                    man_r = float(out_ref_bg) if (bg_en and bg_mode_s == "manual") else None
                    res = integrate_and_quantify(
                        tgt_img, ref_img, ms, tmask, rmask, tchan, rchan,
                        pw, ph,
                        bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                        bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                        manual_tar_bg=man_t, manual_ref_bg=man_r,
                    )
                    # Append background values to outputs so UI updates
                    return (*res, out_tar_bg, out_ref_bg)
                integrate_btn.click(
                    fn=_integrate_callback,
                    inputs=[tgt, ref, masks_state, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan, px_w, px_h, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref],
                    outputs=[integrate_tar_overlay, integrate_ref_overlay, mask_tiff, table, csv_file, tgt_on_and_img, ref_on_and_img, ratio_img, bak_tar, bak_ref],
                )

                # ---------------- Persist settings (Dual) ----------------
                SETTINGS_KEY = "dcq_settings_v1"
                demo.load(
                    fn=None,
                    inputs=[],
                    outputs=[
                        seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu,
                        pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method,
                        rad_in, rad_out, rad_min_obj,
                        tgt_chan, tgt_mask_mode, tgt_sat_limit, tgt_pct, tgt_min_obj,
                        ref_chan, ref_mask_mode, ref_sat_limit, ref_pct, ref_min_obj,
                        px_w, px_h,
                        label_scale,
                    ],
                    js=f"""
                    () => {{
                        try {{
                            const raw = localStorage.getItem('{SETTINGS_KEY}');
                            const d = {{
                                seg_source: 'target', seg_chan: 'gray', diameter: 0, flow_th: 0.4, cellprob_th: 0.0, use_gpu: true,
                                pp_bg_enable: false, pp_bg_mode: 'rolling', pp_bg_radius: 50, pp_dark_pct: 5.0, pp_norm_enable: false, pp_norm_method: 'z-score',
                                rad_in: 0.0, rad_out: 100.0, rad_min_obj: 50,
                                tgt_chan: 'gray', tgt_mask_mode: 'global_percentile', tgt_sat_limit: 254, tgt_pct: 75.0, tgt_min_obj: 50,
                                ref_chan: 'gray', ref_mask_mode: 'global_percentile', ref_sat_limit: 254, ref_pct: 75.0, ref_min_obj: 50,
                                px_w: 1.0, px_h: 1.0,
                                label_scale: {float(LABEL_SCALE)},
                            }};
                            let s = raw ? {{...d, ...JSON.parse(raw)}} : d;
                            const mapChan = (v) => ({{0:'gray',1:'R',2:'G',3:'B'}})[v] ?? v;
                            s.seg_chan = mapChan(s.seg_chan);
                            s.tgt_chan = mapChan(s.tgt_chan);
                            s.ref_chan = mapChan(s.ref_chan);
                            return [
                                s.seg_source,
                                s.seg_chan,
                                s.diameter,
                                s.flow_th,
                                s.cellprob_th,
                                s.use_gpu,
                                s.pp_bg_enable,
                                s.pp_bg_mode,
                                s.pp_bg_radius,
                                s.pp_dark_pct,
                                s.pp_norm_enable,
                                s.pp_norm_method,
                                s.rad_in,
                                s.rad_out,
                                s.rad_min_obj,
                                s.tgt_chan,
                                s.tgt_mask_mode,
                                s.tgt_sat_limit,
                                s.tgt_pct,
                                s.tgt_min_obj,
                                s.ref_chan,
                                s.ref_mask_mode,
                                s.ref_sat_limit,
                                s.ref_pct,
                                s.ref_min_obj,
                                s.px_w,
                                s.px_h,
                                s.label_scale,
                            ];
                        }} catch (e) {{
                            console.warn('Failed to load saved settings:', e);
                            return [
                                'target', 'gray', 0, 0.4, 0.0, true,
                                false, 'rolling', 50, 5.0, false, 'z-score',
                                0.0, 100.0, 50,
                                'gray', 'global_percentile', 254, 75.0, 50,
                                'gray', 'global_percentile', 254, 75.0, 50,
                                1.0, 1.0,
                                {float(LABEL_SCALE)},
                            ];
                        }}
                    }}
                    """,
                )

                def _persist_change(comp, key: str):
                    comp.change(
                        fn=None,
                        inputs=[comp],
                        outputs=[],
                        js=f"""
                        (v) => {{
                            try {{
                                const k = '{SETTINGS_KEY}';
                                const raw = localStorage.getItem(k);
                                const s = raw ? JSON.parse(raw) : {{}};
                                s['{key}'] = v;
                                localStorage.setItem(k, JSON.stringify(s));
                            }} catch (e) {{
                                console.warn('Failed to save setting {key}:', e);
                            }}
                        }}
                        """,
                    )

                for comp, key in [
                    (seg_source, 'seg_source'), (seg_chan, 'seg_chan'), (diameter, 'diameter'), (flow_th, 'flow_th'), (cellprob_th, 'cellprob_th'), (use_gpu, 'use_gpu'),
                    (pp_bg_enable, 'pp_bg_enable'), (pp_bg_mode, 'pp_bg_mode'), (pp_bg_radius, 'pp_bg_radius'), (pp_dark_pct, 'pp_dark_pct'), (pp_norm_enable, 'pp_norm_enable'), (pp_norm_method, 'pp_norm_method'),
                    (rad_in, 'rad_in'), (rad_out, 'rad_out'), (rad_min_obj, 'rad_min_obj'),
                    (tgt_chan, 'tgt_chan'), (tgt_mask_mode, 'tgt_mask_mode'), (tgt_sat_limit, 'tgt_sat_limit'), (tgt_pct, 'tgt_pct'), (tgt_min_obj, 'tgt_min_obj'),
                    (ref_chan, 'ref_chan'), (ref_mask_mode, 'ref_mask_mode'), (ref_sat_limit, 'ref_sat_limit'), (ref_pct, 'ref_pct'), (ref_min_obj, 'ref_min_obj'),
                    (px_w, 'px_w'), (px_h, 'px_h'), (label_scale, 'label_scale'),
                ]:
                    _persist_change(comp, key)

                def _set_label_scale(v: float):
                    global LABEL_SCALE
                    try:
                        LABEL_SCALE = float(v)
                    except Exception:
                        pass
                    return None
                label_scale.change(fn=_set_label_scale, inputs=[label_scale], outputs=[])

                reset_settings.click(
                    fn=None,
                    inputs=[],
                    outputs=[
                        seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu,
                        pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method,
                        rad_in, rad_out, rad_min_obj,
                        tgt_chan, tgt_mask_mode, tgt_sat_limit, tgt_pct, tgt_min_obj,
                        ref_chan, ref_mask_mode, ref_sat_limit, ref_pct, ref_min_obj,
                        px_w, px_h,
                        label_scale,
                    ],
                    js=f"""
                    () => {{
                        try {{
                            localStorage.removeItem('{SETTINGS_KEY}');
                        }} catch (e) {{
                            console.warn('Failed to clear settings:', e);
                        }}
                        alert('Saved settings cleared. Restoring defaults.');
                        return [
                            'target', 'gray', 0, 0.4, 0.0, true,
                            false, 'dark_subtract', 50, 5.0, false, 'min-max',
                            0.0, 100.0, 50,
                            'gray', 'none', 254, 75.0, 50,
                            'gray', 'none', 254, 75.0, 50,
                            1.0, 1.0,
                            {float(LABEL_SCALE)},
                        ];
                    }}
                    """,
                )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch()
