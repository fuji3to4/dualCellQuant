# Stepwise Cellpose-SAM Gradio App
"""
Stepwise pipeline for Cellpose-SAM with Gradio.

Stages:
1. Run Cellpose segmentation.
2. Apply target mask conditions.
3. Apply reference mask conditions.
4. Integrate target & reference masks and compute per-cell statistics.

This allows parameter tuning at each stage before integration.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import tempfile
from skimage import filters, morphology, measure, color
import scipy.ndimage as ndi
from cellpose import models

# -----------------------
# Display/label settings (adjustable via UI)
# -----------------------
LABEL_SCALE: float = 1.8  # default label size multiplier (Linux-friendly)

def _load_font(point_size: int) -> ImageFont.ImageFont:
    """Robust TrueType font loader with common cross-platform fallbacks.

    Tries a list of typical TTF paths; falls back to default bitmap font if none found.
    """
    candidates = [
        # Generic names (might resolve via fontconfig)
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
        "NotoSans-Regular.ttf",
        # Linux common paths
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/ARIALUNI.TTF",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, point_size)
        except Exception:
            continue
    # Last resort
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
    orig_dtype = arr.dtype

    # Drop alpha channel if present (RGBA -> RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Convert to float for downstream processing
    arr = arr.astype(np.float32, copy=False)

    # Normalize to [0, 1] depending on original dtype
    if np.issubdtype(orig_dtype, np.integer):
        # e.g., uint8 -> 255, uint16 -> 65535
        max_val = float(np.iinfo(orig_dtype).max)
        if max_val > 1:
            arr /= max_val
    elif np.issubdtype(orig_dtype, np.floating):
        # If floats come in with a larger range, normalize defensively
        m = np.nanmax(arr)
        if m > 1.0:
            # Avoid divide-by-zero
            arr /= (m if m != 0 else 1.0)
    elif np.issubdtype(orig_dtype, np.bool_):
        # Already 0/1
        pass
    else:
        # Fallback: if something unexpected slips through, clip to [0,1]
        arr = np.clip(arr, 0.0, 1.0)

    return arr

def pil_to_numpy_native(img: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy float32 array without normalizing intensities.

    - Keeps the original numeric scale (e.g., 0-255 for 8-bit, 0-65535 for 16-bit).
    - Drops alpha channel if present.
    - Returns float32 array for numerical operations while preserving value scale.
    """
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.float32, copy=False)

# --- ImageJ向けTIFF保存ヘルパー ---
def save_bool_mask_tiff(mask: np.ndarray, stem: str) -> str:
    """Save boolean mask as 8-bit TIFF (0/255) for ImageJ."""
    arr = (mask.astype(np.uint8) * 255)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.tif")
    Image.fromarray(arr).save(tmp.name)
    return tmp.name

def save_label_tiff(labels: np.ndarray, stem: str) -> str:
    """Save label image as 16-bit TIFF (or 32-bit float if labels > 65535)."""
    maxv = int(labels.max()) if labels.size > 0 else 0
    if maxv <= 65535:
        out = labels.astype(np.uint16, copy=False)
    else:
        out = labels.astype(np.float32, copy=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.tif")
    Image.fromarray(out).save(tmp.name)
    return tmp.name

# 8bit絶対値(0-255)で指定されたSaturation limitを画像ネイティブの0-1に変換
def _to_float_saturation_limit(img: Image.Image, sat_limit_abs_8bit: float) -> float:
    raw = np.array(img)
    # RGBA -> RGB へ
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]

    ref_max = 255.0
    if np.issubdtype(raw.dtype, np.integer):
        native_max = float(np.iinfo(raw.dtype).max)  # 255, 65535 など
    elif np.issubdtype(raw.dtype, np.floating):
        # 浮動小数は0-1想定（PIL 'F'などの場合は値域を見て判断）
        vmax = float(np.nanmax(raw)) if raw.size > 0 else 1.0
        native_max = 1.0 if vmax <= 1.0 else vmax
    else:
        native_max = 1.0

    # 8bit基準 -> ネイティブ絶対値へ変換
    if native_max <= ref_max:
        sat_native = float(np.clip(sat_limit_abs_8bit, 0.0, native_max))
    else:
        scale = native_max / ref_max
        sat_native = float(np.clip(sat_limit_abs_8bit * scale, 0.0, native_max))

    # ネイティブ絶対値 -> 0-1へ
    thr01 = sat_native / (native_max if native_max > 0 else 1.0)
    return float(np.clip(thr01, 0.0, 1.0))


def extract_single_channel(img: np.ndarray, chan) -> np.ndarray:
    """Extract a single channel as float32 grayscale in [0,1].

    chan can be one of:
      - strings: "gray", "R", "G", "B"
      - legacy ints: 0(gray), 1(R), 2(G), 3(B)
    """
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)

    # Normalize channel key
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
# Visualization
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
    # If label scale is zero or negative, skip drawing labels
    try:
        if float(LABEL_SCALE) <= 0.0:
            return img
    except Exception:
        pass
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Larger, more legible font size with scalable factor
    base = max(14, int(min(w, h) * 0.035))
    fsize = max(12, int(base * float(LABEL_SCALE)))
    font = _load_font(fsize)
    props = measure.regionprops(masks)
    for p in props:
        lab = p.label
        cy, cx = p.centroid
        x, y = int(cx), int(cy)
        text = str(lab)
        # Thicker outline for visibility (scale with font size)
        outline_color = (0, 0, 0)
        off = max(2, fsize // 12)
        offsets = [(-off,0),(off,0),(0,-off),(0,off),(-off,-off),(off,off),(-off,off),(off,-off)]
        for dx, dy in offsets:
            draw.text((x+dx, y+dy), text, fill=outline_color, font=font, anchor="mm")
        # Bright fill color to stand out on grayscale background
        fill_color = (255, 255, 0)  # yellow
        draw.text((x, y), text, fill=fill_color, font=font, anchor="mm")
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
    elif mode == "per_cell_percentile":
        return float(np.percentile(cell_pixels, float(np.clip(pct, 0.0, 100.0))))
    else:
        raise ValueError("per_cell_threshold called with invalid mode")


def cleanup_mask(mask: np.ndarray, min_obj_size: int) -> np.ndarray:
    if min_obj_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_obj_size))
        mask = morphology.binary_opening(mask, morphology.disk(1))
    return mask

# -----------------------
# Step 1: Cellpose segmentation
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

    tmp_npy = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_npy, masks)
    tmp_npy.flush(); tmp_npy.close()
    seg_tiff = save_label_tiff(masks, "seg_labels")
    return overlay, tmp_npy.name, seg_tiff, mask_viz, masks

# -----------------------
# Step2: Radial mask step
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
    # 事前計算: 各セルの dmax（中心→境界 最大距離）
    dmax_table = {}
    for lab in labels:
        cell = masks == lab
        p = props_map.get(int(lab))
        if p is None or p.area <= 0:
            continue
        # 形状追従: セル内の距離変換で中心(最大距離)→境界(0)を正規化
        di = ndi.distance_transform_edt(cell)
        dmax = float(di.max())
        if dmax <= 0:
            radial_total |= cell
            continue
        dmax_table[int(lab)] = dmax
        # t は中心0.0, 境界1.0
        t = 1.0 - (di / dmax)
        rin_t = max(0.0, float(inner_pct)) / 100.0
        rout_t = max(0.0, float(outer_pct)) / 100.0
        if rin_t > rout_t:
            rin_t, rout_t = rout_t, rin_t
        # 内側帯域（〜100%まで）
        inside_band = (t >= rin_t) & (t <= min(rout_t, 1.0)) & cell
        radial_cell = inside_band.copy()
        radial_cell = cleanup_mask(radial_cell, int(min_obj_size))
        radial_total |= radial_cell
        radial_labels[radial_cell] = int(lab)

    # 100%超の外側帯域（背景側）は、背景 EDT と最近傍セル割当でラベル拡張
    extra_frac = max(0.0, float(outer_pct) / 100.0 - 1.0)
    if extra_frac > 0.0 and len(dmax_table) > 0:
        fg = masks > 0
        dist_bg, idx = ndi.distance_transform_edt(~fg, return_indices=True)
        # 最近傍のセル画素のラベルを取得
        nearest_label = masks[idx[0], idx[1]]
        max_lab = int(masks.max())
        extra_table = np.zeros(max_lab + 1, dtype=np.float32)
        for lab, dmax in dmax_table.items():
            extra_table[int(lab)] = extra_frac * float(dmax)
        outside_bool = (~fg) & (nearest_label > 0) & (dist_bg > 0) & (dist_bg <= extra_table[nearest_label])
        # 既に内側で塗られた部分は保持、背景側のみを追加
        add_mask = outside_bool & bg
        radial_total |= add_mask
        # ラベル付与（後勝ちしないよう、未設定領域にのみセット）
        to_set = (radial_labels == 0) & add_mask
        radial_labels[to_set] = nearest_label[to_set].astype(radial_labels.dtype)

    # 可視化
    overlay = colorize_overlay((masks>0).astype(np.float32), masks, radial_total)
    overlay = annotate_ids(overlay, masks)
    tmp_bool = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_bool, radial_total)
    tmp_bool.flush(); tmp_bool.close()
    tmp_lbl = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_lbl, radial_labels)
    tmp_lbl.flush(); tmp_lbl.close()
    rad_bool_tiff = save_bool_mask_tiff(radial_total, "radial_mask")
    rad_lbl_tiff = save_label_tiff(radial_labels, "radial_labels")
    return overlay, tmp_bool.name, radial_total, tmp_lbl.name, radial_labels, rad_bool_tiff, rad_lbl_tiff


# -----------------------
# Step 3/4: Apply mask for target or reference
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

    arr = pil_to_numpy(img)
    gray = extract_single_channel(arr, measure_channel)

    # Saturation limit: UIは8bit絶対値(0-255)で入力 -> 0-1へ変換して使用
    sat_thr01 = _to_float_saturation_limit(img, float(sat_limit))
    nonsat = gray < sat_thr01

    mask_total = np.zeros_like(masks, dtype=bool)

    global_mask = None
    if mask_mode in ("global_otsu", "global_percentile"):
        global_mask, _ = global_threshold_mask(gray, nonsat, mask_mode, float(pct), int(min_obj_size))

    labels = np.unique(masks); labels = labels[labels > 0]
    for lab in labels:
        cell = masks == lab
        if roi_labels is not None:
            # ROI が指定されている場合は、対応するラベル領域に限定
            cell = cell & (roi_labels == lab)
        if mask_mode == "none":
            # Noneでも Saturation limit は適用
            mask_cell = cell & nonsat
        elif mask_mode in ("global_otsu", "global_percentile"):
            mask_cell = global_mask & cell
            mask_cell = cleanup_mask(mask_cell, int(min_obj_size))
        elif mask_mode in ("per_cell_otsu", "per_cell_percentile"):
            pool = gray[cell & nonsat]
            th = per_cell_threshold(pool, mask_mode, float(pct))
            base = (gray >= th) & nonsat & cell if not np.isnan(th) else np.zeros_like(cell, dtype=bool)
            mask_cell = cleanup_mask(base, int(min_obj_size))
        else:
            raise ValueError("Invalid mask_mode")
        mask_total |= mask_cell

    overlay = colorize_overlay(gray, masks, mask_total)
    overlay = annotate_ids(overlay, masks)

    tmp_npy = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_npy, mask_total)
    tmp_npy.flush(); tmp_npy.close()
    tiff_path = save_bool_mask_tiff(mask_total, mask_name)
    return overlay, tmp_npy.name, tiff_path, mask_total

# -----------------------
# Helper: apply mask with optional ROI toggle
# -----------------------
def apply_mask_with_roi(
    img: Image.Image,
    masks: np.ndarray,
    measure_channel: int,
    sat_limit: float,
    mask_mode: str,
    pct: float,
    min_obj_size: int,
    use_roi: bool,
    roi_labels: Optional[np.ndarray],
    mask_name: str,
):
    roi = roi_labels if use_roi else None
    return apply_mask(img, masks, measure_channel, sat_limit, mask_mode, pct, min_obj_size, roi, mask_name)


# -----------------------
# Step 5: Integrate masks and quantify
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

):
    if masks is None or tgt_mask is None or ref_mask is None:
        raise ValueError("Run previous steps first.")

    # Visualization arrays (0-1) for overlays/ratio
    tgt_vis = pil_to_numpy(target_img)
    ref_vis = pil_to_numpy(reference_img)
    if tgt_vis.shape[:2] != ref_vis.shape[:2]:
        raise ValueError("Image size mismatch between target and reference")

    # Native arrays (original scale) for measurements to match ImageJ
    tgt_nat = pil_to_numpy_native(target_img)
    ref_nat = pil_to_numpy_native(reference_img)

    tgt_gray_vis = extract_single_channel(tgt_vis, tgt_chan)
    ref_gray_vis = extract_single_channel(ref_vis, ref_chan)
    tgt_gray_nat = extract_single_channel(tgt_nat, tgt_chan)
    ref_gray_nat = extract_single_channel(ref_nat, ref_chan)

    and_mask = tgt_mask & ref_mask
    labels = np.unique(masks); labels = labels[labels > 0]
    vis_union = np.zeros_like(masks, dtype=bool)
    # Area conversion (um^2 per pixel)
    try:
        px_area_um2 = float(pixel_width_um) * float(pixel_height_um)
    except Exception:
        px_area_um2 = 1.0

    rows = []
    for lab in labels:
        cell = masks == lab
        idx = and_mask & cell
        vis_union |= idx
        area_cell_px = int(cell.sum())
        area_and_px = int(idx.sum())
        area_cell_um2 = float(area_cell_px) * px_area_um2
        area_and_um2 = float(area_and_px) * px_area_um2
        if area_and_px == 0:
            mean_t_mem = std_t_mem = sum_t_mem = mean_r_mem = std_r_mem = sum_r_mem = ratio_mean = ratio_sum = ratio_std = np.nan
        else:
            # Use native intensities for statistics (ImageJ-like)
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
        # Whole-cell stats also in native intensity scale
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
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_csv.name, index=False)

    overlay = colorize_overlay(tgt_gray_vis, masks, and_mask)
    overlay = annotate_ids(overlay, masks)
   

    tmp_npy = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_npy, and_mask)
    tmp_npy.flush(); tmp_npy.close()
    and_tiff = save_bool_mask_tiff(and_mask, "and_mask")

    # --- 比率画像の生成（T/R） ---
    # refが0のところはNaNにして割り算回避
    valid = (ref_gray_vis > 0)
    ratio_img = np.full_like(tgt_gray_vis, np.nan, dtype=np.float32)
    ratio_img[valid] = tgt_gray_vis[valid] / ref_gray_vis[valid]

    # 表示は AND マスク内に限定（外側は黒に落とす）
    ratio_masked = np.where(and_mask, ratio_img, np.nan)

    # 可視化のために頑健な正規化（外れ値対策で1〜99パーセンタイル）
    finite_vals = ratio_masked[np.isfinite(ratio_masked)]
    if finite_vals.size > 0:
        vmin = np.percentile(finite_vals, 1.0)
        vmax = np.percentile(finite_vals, 99.0)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        ratio_norm = (ratio_masked - vmin) / (vmax - vmin)
    else:
        ratio_norm = np.zeros_like(ratio_masked, dtype=np.float32)

    # NaN→0 にしてグレイスケール
    ratio_for_viz = np.nan_to_num(ratio_norm, nan=0.0, posinf=0.0, neginf=0.0)
    #ratio_overlay = colorize_overlay(ratio_for_viz, masks, and_mask) #赤：輪郭、緑：ANDマスク境界
    ratio_overlay = Image.fromarray((ratio_for_viz * 255).astype(np.uint8))  # マスク線なし


    # 生の比率配列を保存（NaN含む）
    tmp_ratio_npy = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_ratio_npy, ratio_img)
    tmp_ratio_npy.flush(); tmp_ratio_npy.close()

    return overlay, tmp_npy.name, and_tiff, df, tmp_csv.name, ratio_overlay, tmp_ratio_npy.name


# -----------------------
# UI
# -----------------------

def build_ui():
    with gr.Blocks(title="DualCellQuant") as demo:
        gr.Markdown(
            """
            # 🔬 **DualCellQuant**
            *Segment, filter, and compare cells across two fluorescence channels*
            1. **Run Cellpose-SAM** to obtain segmentation masks.
            2. **Build Radial mask** (optional).
            3. **Apply Target mask** conditions.
            4. **Apply Reference mask** conditions.
            5. **Integrate** Target & Reference masks and view results.
            Each step can be rerun to tune parameters before integration.
            """
        )
        masks_state = gr.State()
        radial_mask_state = gr.State()           # bool mask
        radial_label_state = gr.State()          # labeled mask
        tgt_mask_state = gr.State()
        ref_mask_state = gr.State()
        
        with gr.Row():
            with gr.Column():
                tgt = gr.Image(type="pil", label="Target image", image_mode="RGB", width=600)
                ref = gr.Image(type="pil", label="Reference image", image_mode="RGB", width=600)

                with gr.Accordion("Segmentation params", open=False):
                    seg_source = gr.Radio(["target","reference"], value="target", label="Segment on")
                    seg_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Segmentation channel")
                    diameter = gr.Slider(0, 200, value=0, step=1, label="Diameter (px, 0=auto)")
                    flow_th = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                    cellprob_th = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")
                    use_gpu = gr.Checkbox(value=True, label="Use GPU if available")
                run_seg_btn = gr.Button("1. Run Cellpose")
                seg_overlay = gr.Image(type="pil", label="Segmentation overlay",width=600)
                mask_img = gr.Image(type="pil", label="Segmentation label image",width=600)
                # seg_npy_state = gr.File(label="Download masks (.npy)")
                seg_npy_state = gr.State()
                seg_tiff_file = gr.File(label="Download masks (label TIFF)")
                # Hidden holders to capture .npy paths without showing download widgets
                
                with gr.Accordion("Radial mask (optional)", open=False):
                    rad_in = gr.Slider(0.0, 120.0, value=0.0, step=1.0, label="Radial inner % (0=中心)")
                    rad_out = gr.Slider(0.0, 120.0, value=100.0, step=1.0, label="Radial outer % (100=境界)")
                    rad_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                run_rad_btn = gr.Button("2. Build Radial mask")
                rad_overlay = gr.Image(type="pil", label="Radial mask overlay", width=600)
                # rad_npy_state = gr.File(label="Download radial mask (bool .npy)")
                rad_npy_state = gr.State()
                # rad_lbl_npy_stat = gr.File(label="Download radial labels (.npy)")
                rad_lbl_npy_state = gr.State()
                rad_tiff = gr.File(label="Download radial mask (TIFF)")
                rad_lbl_tiff = gr.File(label="Download radial labels (label TIFF)")

                use_radial_roi_tgt = gr.Checkbox(value=False, label="Use Radial ROI for Target mask")
                use_radial_roi_ref = gr.Checkbox(value=False, label="Use Radial ROI for Reference mask")

                with gr.Accordion("Target mask", open=False):
                    tgt_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Target channel")
                    tgt_mask_mode = gr.Radio(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                    # 8bit絶対値（0-255）指定に変更
                    tgt_sat_limit = gr.Slider(0, 255, value=254, step=1, label="Saturation limit (abs, 8-bit scale)")
                    tgt_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                    tgt_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                run_tgt_btn = gr.Button("3. Apply Target mask")
                tgt_overlay = gr.Image(type="pil", label="Target mask overlay",width=600)
                # tgt_npy_state = gr.File(label="Download target mask (.npy)")
                tgt_npy_state = gr.State()
                tgt_tiff = gr.File(label="Download target mask (TIFF)")
                

                with gr.Accordion("Reference mask", open=False):
                    ref_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Reference channel")
                    ref_mask_mode = gr.Radio(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                    # 8bit絶対値（0-255）指定に変更
                    ref_sat_limit = gr.Slider(0, 255, value=254, step=1, label="Saturation limit (abs, 8-bit scale)")
                    ref_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                    ref_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                run_ref_btn = gr.Button("4. Apply Reference mask")
                ref_overlay = gr.Image(type="pil", label="Reference mask overlay",width=600)
                # ref_npy_state = gr.File(label="Download reference mask (.npy)")
                ref_npy_state = gr.State()
                ref_tiff = gr.File(label="Download reference mask (TIFF)")
                

                integrate_btn = gr.Button("5. Integrate & Quantify")
                # Pixel size inputs for ImageJ-compatible area (µm^2)
                px_w = gr.Number(value=1.0, label="Pixel width (µm)")
                px_h = gr.Number(value=1.0, label="Pixel height (µm)")

                # Label size control (helpful for Linux servers)
                label_scale = gr.Slider(0.0, 5.0, value=float(LABEL_SCALE), step=0.1, label="Label size scale (0=hidden)")

                final_overlay = gr.Image(type="pil", label="Final overlay (AND mask)",width=600)
                
                # and_npy_state = gr.File(label="Download AND mask (.npy)")
                and_npy_state = gr.State()
                mask_tiff = gr.File(label="Download AND mask (TIFF)")
                table = gr.Dataframe(label="Per-cell intensities & ratios", interactive=False)
                csv_file = gr.File(label="Download CSV")
                
                ratio_img = gr.Image(type="pil", label="Ratio (Target/Reference) on AND mask", width=600)
                # ratio_npy_state = gr.File(label="Download ratio (T_over_R) (.npy)")                
                ratio_npy_state = gr.State()

                # Reset saved UI settings
                reset_settings = gr.Button("Reset saved settings")

        run_seg_btn.click(
            fn=run_segmentation,
            inputs=[tgt, ref, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu],
            outputs=[seg_overlay, seg_npy_state, seg_tiff_file, mask_img, masks_state],
        )
        run_rad_btn.click(
            fn=radial_mask,
            inputs=[masks_state, rad_in, rad_out, rad_min_obj],
            outputs=[rad_overlay, rad_npy_state, radial_mask_state, rad_lbl_npy_state, radial_label_state, rad_tiff, rad_lbl_tiff],
        )
        # Wrapper that passes ROI based on checkbox
        run_tgt_btn.click(
            fn=apply_mask_with_roi,
            inputs=[tgt, masks_state, tgt_chan, tgt_sat_limit, tgt_mask_mode, tgt_pct, tgt_min_obj, use_radial_roi_tgt, radial_label_state, gr.State("target_mask")],
            outputs=[tgt_overlay, tgt_npy_state, tgt_tiff, tgt_mask_state],
        )
        run_ref_btn.click(
            fn=apply_mask_with_roi,
            inputs=[ref, masks_state, ref_chan, ref_sat_limit, ref_mask_mode, ref_pct, ref_min_obj, use_radial_roi_ref, radial_label_state, gr.State("reference_mask")],
            outputs=[ref_overlay, ref_npy_state, ref_tiff, ref_mask_state],
        )
        integrate_btn.click(
            fn=integrate_and_quantify,
            inputs=[tgt, ref, masks_state, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan, px_w, px_h],
            outputs=[final_overlay, and_npy_state, mask_tiff, table, csv_file, ratio_img, ratio_npy_state],
        )

        # -----------------------
        # Persist UI settings in browser localStorage
        # -----------------------
        SETTINGS_KEY = "dcq_settings_v1"

        # Restore saved settings on app load
        demo.load(
            fn=None,
            inputs=[],
            outputs=[
                seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu,
                rad_in, rad_out, rad_min_obj,
                use_radial_roi_tgt, use_radial_roi_ref,
                tgt_chan, tgt_mask_mode, tgt_sat_limit, tgt_pct, tgt_min_obj,
                ref_chan, ref_mask_mode, ref_sat_limit, ref_pct, ref_min_obj,
                px_w, px_h,
                label_scale,
            ],
            js=f"""
            () => {{
                try {{
                    const raw = localStorage.getItem('{SETTINGS_KEY}');
                    // Defaults aligned with component value= in Python UI
                    const d = {{
                        seg_source: 'target', seg_chan: 'gray', diameter: 0, flow_th: 0.4, cellprob_th: 0.0, use_gpu: true,
                        rad_in: 0.0, rad_out: 100.0, rad_min_obj: 50,
                        use_radial_roi_tgt: false, use_radial_roi_ref: false,
                        tgt_chan: 'gray', tgt_mask_mode: 'global_percentile', tgt_sat_limit: 254, tgt_pct: 75.0, tgt_min_obj: 50,
                        ref_chan: 'gray', ref_mask_mode: 'global_percentile', ref_sat_limit: 254, ref_pct: 75.0, ref_min_obj: 50,
                        px_w: 1.0, px_h: 1.0,
                        label_scale: {float(LABEL_SCALE)},
                    }};
                    let s = raw ? {{...d, ...JSON.parse(raw)}} : d;
                    // Backward compatibility: map numeric channels to strings if needed
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
                        s.rad_in,
                        s.rad_out,
                        s.rad_min_obj,
                        s.use_radial_roi_tgt,
                        s.use_radial_roi_ref,
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
                    // Fallback to defaults if parsing/storage fails
                    return [
                        'target', 'gray', 0, 0.4, 0.0, true,
                        0.0, 100.0, 50,
                        false, false,
                        'gray', 'global_percentile', 254, 75.0, 50,
                        'gray', 'global_percentile', 254, 75.0, 50,
                        1.0, 1.0,
                        {float(LABEL_SCALE)},
                    ];
                }}
            }}
            """,
        )

        # Helper to register change->save for a component key
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

        # Register persistence for target/reference and segmentation params
        _persist_change(seg_source, 'seg_source')
        _persist_change(seg_chan, 'seg_chan')
        _persist_change(diameter, 'diameter')
        _persist_change(flow_th, 'flow_th')
        _persist_change(cellprob_th, 'cellprob_th')
        _persist_change(use_gpu, 'use_gpu')

        _persist_change(rad_in, 'rad_in')
        _persist_change(rad_out, 'rad_out')
        _persist_change(rad_min_obj, 'rad_min_obj')
        _persist_change(use_radial_roi_tgt, 'use_radial_roi_tgt')
        _persist_change(use_radial_roi_ref, 'use_radial_roi_ref')

        _persist_change(tgt_chan, 'tgt_chan')
        _persist_change(tgt_mask_mode, 'tgt_mask_mode')
        _persist_change(tgt_sat_limit, 'tgt_sat_limit')
        _persist_change(tgt_pct, 'tgt_pct')
        _persist_change(tgt_min_obj, 'tgt_min_obj')

        _persist_change(ref_chan, 'ref_chan')
        _persist_change(ref_mask_mode, 'ref_mask_mode')
        _persist_change(ref_sat_limit, 'ref_sat_limit')
        _persist_change(ref_pct, 'ref_pct')
        _persist_change(ref_min_obj, 'ref_min_obj')
        # Persist pixel size settings
        _persist_change(px_w, 'px_w')
        _persist_change(px_h, 'px_h')
        _persist_change(label_scale, 'label_scale')

        # Update global label scale when slider changes
        def _set_label_scale(v: float):
            global LABEL_SCALE
            try:
                LABEL_SCALE = float(v)
            except Exception:
                pass
            return None
        label_scale.change(fn=_set_label_scale, inputs=[label_scale], outputs=[])

        # Reset button clears stored settings
        reset_settings.click(
            fn=None,
            inputs=[],
            outputs=[
                seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu,
                rad_in, rad_out, rad_min_obj,
                use_radial_roi_tgt, use_radial_roi_ref,
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
                    0.0, 100.0, 50,
                    false, false,
                    'gray', 'global_percentile', 254, 75.0, 50,
                    'gray', 'global_percentile', 254, 75.0, 50,
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
