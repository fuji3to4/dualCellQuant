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
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3].astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def extract_single_channel(img: np.ndarray, chan: int) -> np.ndarray:
    if img.ndim == 2:
        return img
    if chan == 0:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    elif chan in (1, 2, 3):
        return img[:, :, chan - 1].astype(np.float32)
    else:
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
    draw = ImageDraw.Draw(img)
    w, h = img.size
    fsize = max(10, int(min(w, h) * 0.02))
    try:
        font = ImageFont.truetype("arial.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()
    props = measure.regionprops(masks)
    for p in props:
        lab = p.label
        cy, cx = p.centroid
        x, y = int(cx), int(cy)
        text = str(lab)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            draw.text((x+dx, y+dy), text, fill=(255,255,255), font=font, anchor="mm")
        draw.text((x, y), text, fill=(0,0,0), font=font, anchor="mm")
    return img

# -----------------------
# Threshold utilities
# -----------------------

def global_threshold_mask(img_gray: np.ndarray, sat_limit: float, mode: str, pct: float, min_obj_size: int) -> Tuple[np.ndarray, float]:
    nonsat = img_gray < sat_limit
    if mode == "global_otsu":
        valid = img_gray[nonsat]
        if valid.size == 0:
            raise ValueError("All pixels are saturated; lower the saturation limit")
        th = float(filters.threshold_otsu(valid))
    elif mode == "global_percentile":
        valid = img_gray[nonsat]
        if valid.size == 0:
            raise ValueError("All pixels are saturated; lower the saturation limit")
        th = float(np.percentile(valid, float(np.clip(pct, 0.0, 100.0))))
    else:
        raise ValueError("global_threshold_mask called with invalid mode")
    mask = (img_gray >= th) & nonsat
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
    return overlay, tmp_npy.name, mask_viz, masks

# -----------------------
# Step 2/3: Apply mask for target or reference
# -----------------------

def apply_mask(
    img: Image.Image,
    masks: np.ndarray,
    measure_channel: int,
    sat_limit: float,
    mask_mode: str,
    pct: float,
    min_obj_size: int,
):
    if masks is None:
        raise ValueError("Segmentation masks not provided. Run segmentation first.")

    arr = pil_to_numpy(img)
    gray = extract_single_channel(arr, measure_channel)
    nonsat = gray < float(sat_limit)
    mask_total = np.zeros_like(masks, dtype=bool)

    global_mask = None
    if mask_mode in ("global_otsu", "global_percentile"):
        global_mask, _ = global_threshold_mask(gray, float(sat_limit), mask_mode, float(pct), int(min_obj_size))

    labels = np.unique(masks); labels = labels[labels > 0]
    for lab in labels:
        cell = masks == lab
        if mask_mode == "none":
            mask_cell = cell
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
    return overlay, tmp_npy.name, mask_total

# -----------------------
# Step 4: Integrate masks and quantify
# -----------------------

def integrate_and_quantify(
    target_img: Image.Image,
    reference_img: Image.Image,
    masks: np.ndarray,
    tgt_mask: np.ndarray,
    ref_mask: np.ndarray,
    tgt_chan: int,
    ref_chan: int,
):
    if masks is None or tgt_mask is None or ref_mask is None:
        raise ValueError("Run previous steps first.")

    tgt = pil_to_numpy(target_img)
    ref = pil_to_numpy(reference_img)
    if tgt.shape[:2] != ref.shape[:2]:
        raise ValueError("Image size mismatch between target and reference")

    tgt_gray = extract_single_channel(tgt, tgt_chan)
    ref_gray = extract_single_channel(ref, ref_chan)

    and_mask = tgt_mask & ref_mask
    labels = np.unique(masks); labels = labels[labels > 0]
    vis_union = np.zeros_like(masks, dtype=bool)
    rows = []
    for lab in labels:
        cell = masks == lab
        idx = and_mask & cell
        vis_union |= idx
        area_cell = int(cell.sum())
        area_and = int(idx.sum())
        if area_and == 0:
            mean_t_mem = sum_t_mem = mean_r_mem = sum_r_mem = ratio_mean = ratio_sum = ratio_std = np.nan
        else:
            mean_t_mem = float(np.mean(tgt_gray[idx]))
            sum_t_mem = float(np.sum(tgt_gray[idx]))
            mean_r_mem = float(np.mean(ref_gray[idx]))
            sum_r_mem = float(np.sum(ref_gray[idx]))
            if np.all(ref_gray[idx] > 0):
                ratio_vals = tgt_gray[idx] / ref_gray[idx]
                ratio_mean = float(np.mean(ratio_vals))
                ratio_std = float(np.std(ratio_vals))
                ratio_sum = float(np.sum(ratio_vals))
            else:
                ratio_mean = ratio_std = ratio_sum = np.nan
        mean_t_whole = float(np.mean(tgt_gray[cell]))
        sum_t_whole = float(np.sum(tgt_gray[cell]))
        mean_r_whole = float(np.mean(ref_gray[cell]))
        sum_r_whole = float(np.sum(ref_gray[cell]))
        rows.append({
            "label": int(lab),
            "area_cell_px": area_cell,
            "area_and_px": area_and,
            "mean_target_on_mask": mean_t_mem,
            "sum_target_on_mask": sum_t_mem,
            "mean_reference_on_mask": mean_r_mem,
            "sum_reference_on_mask": sum_r_mem,
            "ratio_sum_T_over_R": ratio_sum,
            "ratio_mean_T_over_R": ratio_mean,
            "ratio_std_T_over_R": ratio_std,
            "mean_target_whole": mean_t_whole,
            "sum_target_whole": sum_t_whole,
            "mean_reference_whole": mean_r_whole,
            "sum_reference_whole": sum_r_whole,
        })

    df = pd.DataFrame(rows).sort_values("label")
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_csv.name, index=False)

    overlay = colorize_overlay(tgt_gray, masks, and_mask)
    overlay = annotate_ids(overlay, masks)
   

    tmp_npy = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_npy, and_mask)
    tmp_npy.flush(); tmp_npy.close()

    return overlay, tmp_npy.name, df, tmp_csv.name

# -----------------------
# UI
# -----------------------

def build_ui():
    with gr.Blocks(title="Cellpose2Quant: Stepwise 2-channel Cell Image Quantification using Cellpose-SAM") as demo:
        gr.Markdown(
            """
            # Cellpose-SAM: Stepwise 2-channel Quantification
            1. **Run Cellpose** to obtain segmentation masks.
            2. **Apply Target mask** conditions.
            3. **Apply Reference mask** conditions.
            4. **Integrate** Target & Reference masks and view results.
            Each step can be rerun to tune parameters before integration.
            """
        )
        masks_state = gr.State()
        tgt_mask_state = gr.State()
        ref_mask_state = gr.State()

        with gr.Row():
            with gr.Column():
                tgt = gr.Image(type="pil", label="Target image", image_mode="L",width=600)
                ref = gr.Image(type="pil", label="Reference image", image_mode="L",width=600)

                with gr.Accordion("Segmentation params", open=False):
                    seg_source = gr.Radio(["target","reference"], value="target", label="Segment on")
                    seg_chan = gr.Radio([0,1,2,3], value=0, label="Segmentation channel")
                    diameter = gr.Slider(0, 200, value=0, step=1, label="Diameter (px, 0=auto)")
                    flow_th = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                    cellprob_th = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")
                    use_gpu = gr.Checkbox(value=True, label="Use GPU if available")
                run_seg_btn = gr.Button("1. Run Cellpose")
                seg_overlay = gr.Image(type="pil", label="Segmentation overlay",width=600)
                mask_img = gr.Image(type="pil", label="Segmentation label image",width=600)
                seg_file = gr.File(label="Download masks (.npy)")

                with gr.Accordion("Target mask", open=False):
                    tgt_chan = gr.Radio([0,1,2,3], value=0, label="Target channel")
                    tgt_mask_mode = gr.Radio(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                    tgt_sat_limit = gr.Slider(0.80, 1.0, value=0.98, step=0.001, label="Saturation limit (Target<limit)")
                    tgt_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                    tgt_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                run_tgt_btn = gr.Button("2. Apply Target mask")
                tgt_overlay = gr.Image(type="pil", label="Target mask overlay",width=600)
                tgt_file = gr.File(label="Download target mask (.npy)")

                with gr.Accordion("Reference mask", open=False):
                    ref_chan = gr.Radio([0,1,2,3], value=0, label="Reference channel")
                    ref_mask_mode = gr.Radio(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                    ref_sat_limit = gr.Slider(0.80, 1.0, value=0.98, step=0.001, label="Saturation limit (Reference<limit)")
                    ref_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                    ref_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                run_ref_btn = gr.Button("3. Apply Reference mask")
                ref_overlay = gr.Image(type="pil", label="Reference mask overlay",width=600)
                ref_file = gr.File(label="Download reference mask (.npy)")

                integrate_btn = gr.Button("4. Integrate & Quantify")

                final_overlay = gr.Image(type="pil", label="Final overlay (AND mask)",width=600)
                
                mask_npy = gr.File(label="Download AND mask (.npy)")
                table = gr.Dataframe(label="Per-cell intensities & ratios", interactive=False)
                csv_file = gr.File(label="Download CSV")

        run_seg_btn.click(
            fn=run_segmentation,
            inputs=[tgt, ref, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu],
            outputs=[seg_overlay, seg_file, mask_img, masks_state],
        )
        run_tgt_btn.click(
            fn=apply_mask,
            inputs=[tgt, masks_state, tgt_chan, tgt_sat_limit, tgt_mask_mode, tgt_pct, tgt_min_obj],
            outputs=[tgt_overlay, tgt_file, tgt_mask_state],
        )
        run_ref_btn.click(
            fn=apply_mask,
            inputs=[ref, masks_state, ref_chan, ref_sat_limit, ref_mask_mode, ref_pct, ref_min_obj],
            outputs=[ref_overlay, ref_file, ref_mask_state],
        )
        integrate_btn.click(
            fn=integrate_and_quantify,
            inputs=[tgt, ref, masks_state, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan],
            outputs=[final_overlay, mask_npy, table, csv_file],
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch()
