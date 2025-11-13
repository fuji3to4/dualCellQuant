"""
Gradio UI for DualCellQuant.

Two tabs:
1. 📋 Step-by-Step: Individual control over steps 1-4
2. ⚡ Radial Profile: One-click pipeline for steps 1→2→3→5→6

Settings are shared via localStorage.
"""

import gradio as gr
import tempfile
import pandas as pd
import numpy as np

from dualcellquant import *
from dualcellquant.ui_callbacks_stepbystep import create_stepbystep_callbacks
from dualcellquant.ui_callbacks_radialprofile import create_radialprofile_callbacks
import dualcellquant as dcq

# localStorage key for settings persistence
SETTINGS_KEY = "dcq_settings_v1"

def build_ui():
    with gr.Blocks(title="DualCellQuant") as demo:
        gr.Markdown(
            """
            # 🔬 **DualCellQuant**
            *Segment, filter, and compare cells across two fluorescence channels*


            1. Run Cellpose-SAM to obtain segmentation masks.
            2. Apply Target/Reference masks.
            3. Integrate (Preprocess applied only here) and view results.
            4. Build Radial mask & Quantify
            
            **📋 Step-by-Step**: Detailed control (Steps 1→2→3→4)  
            **⚡ Radial Profile**: 
            One-click pipeline (Steps 1→2→3) with advanced radial profile and peak difference analysis.
            """
        )
        
        # Shared state variables
        masks_state = gr.State()
        radial_mask_state = gr.State()
        radial_label_state = gr.State()
        tgt_mask_state = gr.State()
        ref_mask_state = gr.State()
        quant_df_state = gr.State()
        radial_quant_df_state = gr.State()
        prof_cache_df_state = gr.State()
        prof_cache_csv_state = gr.State()
        prof_cache_plot_state = gr.State()
        prof_cache_params_state = gr.State()
        peak_diff_state = gr.State()
        
        with gr.Tabs() as tabs:
            # ==================== Tab 1: Step-by-Step (Steps 1-4) ====================
            with gr.TabItem("📋 Step-by-Step", id=0) as tab_stepbystep:
                masks_state = gr.State()
                radial_mask_state = gr.State()
                radial_label_state = gr.State()
                tgt_mask_state = gr.State()
                ref_mask_state = gr.State()
                quant_df_state = gr.State()  # Store Step 3 quantification DataFrame (for peak analysis)
                radial_quant_df_state = gr.State()  # Store radial quantification DataFrame
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Input Images")
                        with gr.Row():
                            tgt = gr.Image(type="pil", label="Target image", image_mode="RGB", width=600)
                            ref = gr.Image(type="pil", label="Reference image", image_mode="RGB", width=600)
                        
                        with gr.Accordion("Settings", open=False):
                            reset_settings = gr.Button("Reset Settings",scale=1)
                            label_scale = gr.Slider(0.0, 5.0, value=float(LABEL_SCALE), step=0.1, label="Label size scale (0=hidden)")
                            # hidden state used to pass JS-initialized value into Python on load
                            label_scale_init_state = gr.State()
                        gr.Markdown("## 1. Run Cellpose-SAM Segmentation")
                        with gr.Accordion("Segmentation params", open=True):
                            seg_source = gr.Radio(["target","reference"], value="target", label="Segment on")
                            seg_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Segmentation channel")
                            diameter = gr.Slider(0, 200, value=0, step=1, label="Diameter (px, 0=auto)")
                            flow_th = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                            cellprob_th = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")
                            use_gpu = gr.Checkbox(value=True, label="Use GPU if available")
                            seg_drop_edge = gr.Checkbox(value=True, label="Exclude edge-touching labels")           
                                
                        run_seg_btn = gr.Button("1. Run Cellpose")
                        
                        with gr.Row():
                            seg_overlay = gr.Image(type="pil", label="Segmentation overlay", width=600)
                            mask_img = gr.Image(type="pil", label="Segmentation label image", width=600)
                        seg_tiff_file = gr.File(label="Download masks (label TIFF)")
                        with gr.Accordion("Maintain IDs across frames", open=False):
                            prev_label_tiff = gr.File(label="Previous labels (label TIFF)")
                            iou_match_th = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="IoU threshold for ID match")
                            relabel_btn = gr.Button("Relabel current masks to previous IDs")
                        # Radial mask controls are moved to the bottom (after Integrate)
                        gr.Markdown("## 2. Apply Masks")
                        with gr.Accordion("Apply mask", open=True):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**Target mask settings**")
                                    tgt_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Target channel")
                                    tgt_mask_mode = gr.Dropdown(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="none", label="Masking mode")
                                    tgt_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                                    tgt_sat_limit = gr.Slider(0, 255, value=254, step=1, label="Saturation limit (abs, 8-bit scale)")
                                    tgt_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                                with gr.Column():
                                    gr.Markdown("**Reference mask settings**")
                                    ref_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Reference channel")
                                    ref_mask_mode = gr.Dropdown(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="none", label="Masking mode")
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
                        gr.Markdown("## 3. Integrate & Quantify")
                        with gr.Accordion("Integrate & Quantify", open=True):
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
                                ratio_eps = gr.Number(value=1e-6, label="Ratio epsilon ε (use (T+ε)/(R+ε))", scale=1)
                            with gr.Column():
                                with gr.Row():
                                    px_w = gr.Number(value=1.0, label="Pixel width (µm)", scale=1)
                                    px_h = gr.Number(value=1.0, label="Pixel height (µm)", scale=1)
                        
                        integrate_btn = gr.Button("3. Integrate & Quantify")
                        
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
                            
                        gr.Markdown("## 4. Build Radial Mask & Quantify")
                        # Radial mask section moved here (after integration)
                        with gr.Accordion("Radial mask (optional, after integration)", open=True):
                            rad_in = gr.Slider(0.0, 150.0, value=0.0, step=1.0, label="Radial inner % (0=中心)")
                            rad_out = gr.Slider(0.0, 150.0, value=100.0, step=1.0, label="Radial outer % (100=境界)")
                            rad_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")

                        run_rad_btn = gr.Button("4. Build Radial mask & Quantify")
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
                          
                        gr.Markdown("## (Adv. Radial Intensity Profile)")  
                        with gr.Accordion("Adv. Radial Intensity Profile", open=False):

                            # Radial profile (banded) section
                            with gr.Accordion("5. Radial intensity profile", open=True):
                                prof_start = gr.Number(value=0.0, label="Start %", scale=1)
                                prof_end = gr.Number(value=150.0, label="End %", scale=1)
                                prof_window_size = gr.Number(value=5.0, label="Window size (%)", scale=1)
                                prof_window_step = gr.Number(value=2.0, label="Window moving step (%)", scale=1)
                                # Deprecated (moving average). Hidden to reduce clutter; SG is used for plotting/detection.
                                prof_smoothing = gr.Number(value=1, label="[deprecated] Plot smoothing (moving avg bins)", scale=1, visible=False)

                                
                            # Cache states for radial profile results (computed by 6.)
                            prof_cache_df_state = gr.State()
                            prof_cache_csv_state = gr.State()
                            prof_cache_plot_state = gr.State()
                            prof_cache_params_state = gr.State()
                            peak_diff_state = gr.State()  # Store peak difference DataFrame for plot overlay
                            
                            run_prof_btn = gr.Button("5. Compute Radial profile")
                            
                            with gr.Tabs():
                                with gr.TabItem("RAW"):
                                    profile_table_raw = gr.Dataframe(label="Radial profile (RAW)", interactive=False, pinned_columns=1)
                                    profile_csv = gr.File(label="Download RAW profile CSV")
                                with gr.TabItem("SG (smoothed)"):
                                    profile_table_sg = gr.Dataframe(label="Radial profile (SG-applied)", interactive=False, pinned_columns=1)
                                    profile_csv_sg = gr.File(label="Download SG profile CSV")
                            # SG controls (applied on update)
                            with gr.Row():
                                sg_window = gr.Number(value=5, label="SG window (odd)", precision=0)
                                sg_poly = gr.Number(value=2, label="SG poly", precision=0)
                            # prof_label = "All"
                            prof_show_err = gr.Checkbox(value=True, label="Show error bars (SEM)")
                            profile_show_ratio = gr.Checkbox(value=True, label="Show T/R ratio curve")
                            run_prof_single_btn = gr.Button("Changed label, update profile",)
                            
                            profile_plot = gr.Image(type="pil", label="Radial profile (3-col grid)", width=900)
                            
                            gr.Markdown("## 6. Radial Peak Difference Analysis")
                            # Peak difference section
                            with gr.Accordion("6. Peak difference analysis", open=True):
                                peak_min_pct = gr.Number(value=60.0, label="Min center_pct (%)", scale=1)
                                peak_max_pct = gr.Number(value=120.0, label="Max center_pct (%)", scale=1)
                                peak_algo = gr.Dropdown([
                                    "first_local_top",
                                    "global_max",
                                ], value="first_local_top", label="Peak algorithm")
                                peak_slope_eps_rel = gr.Number(value=0.001, label="Slope eps (relative)")
                            
                            run_peak_diff_btn = gr.Button("6. Compute Peak Differences")
                            
                            peak_diff_table = gr.Dataframe(label="Peak difference per label", interactive=False, pinned_columns=1)
                            peak_diff_csv = gr.File(label="Download peak difference CSV")

                # ==================== Wire up Step-by-Step callbacks ====================
                stepbystep_components = {
                    'tgt': tgt, 'ref': ref,
                    'seg_source': seg_source, 'seg_chan': seg_chan, 'diameter': diameter,
                    'flow_th': flow_th, 'cellprob_th': cellprob_th, 'use_gpu': use_gpu, 'seg_drop_edge': seg_drop_edge,
                    'run_seg_btn': run_seg_btn, 'seg_overlay': seg_overlay, 'seg_tiff_file': seg_tiff_file,
                    'mask_img': mask_img, 'masks_state': masks_state,
                    'prev_label_tiff': prev_label_tiff, 'iou_match_th': iou_match_th, 'relabel_btn': relabel_btn,
                    'prof_cache_df_state': prof_cache_df_state, 'prof_cache_csv_state': prof_cache_csv_state,
                    'prof_cache_plot_state': prof_cache_plot_state, 'prof_cache_params_state': prof_cache_params_state,
                    'tgt_chan': tgt_chan, 'tgt_mask_mode': tgt_mask_mode, 'tgt_pct': tgt_pct,
                    'tgt_sat_limit': tgt_sat_limit, 'tgt_min_obj': tgt_min_obj,
                    'ref_chan': ref_chan, 'ref_mask_mode': ref_mask_mode, 'ref_pct': ref_pct,
                    'ref_sat_limit': ref_sat_limit, 'ref_min_obj': ref_min_obj,
                    'run_tgt_btn': run_tgt_btn, 'tgt_overlay': tgt_overlay, 'tgt_tiff': tgt_tiff,
                    'tgt_mask_state': tgt_mask_state, 'ref_overlay': ref_overlay, 'ref_tiff': ref_tiff,
                    'ref_mask_state': ref_mask_state,
                    'pp_bg_enable': pp_bg_enable, 'pp_bg_mode': pp_bg_mode, 'pp_bg_radius': pp_bg_radius,
                    'pp_dark_pct': pp_dark_pct, 'pp_norm_enable': pp_norm_enable, 'pp_norm_method': pp_norm_method,
                    'bak_tar': bak_tar, 'bak_ref': bak_ref, 'ratio_eps': ratio_eps,
                    'px_w': px_w, 'px_h': px_h,
                    'integrate_btn': integrate_btn, 'integrate_tar_overlay': integrate_tar_overlay,
                    'integrate_ref_overlay': integrate_ref_overlay, 'mask_tiff': mask_tiff,
                    'table': table, 'csv_file': csv_file, 'tgt_on_and_img': tgt_on_and_img,
                    'ref_on_and_img': ref_on_and_img, 'ratio_img': ratio_img, 'quant_df_state': quant_df_state,
                    'rad_in': rad_in, 'rad_out': rad_out, 'rad_min_obj': rad_min_obj,
                    'run_rad_btn': run_rad_btn, 'rad_overlay': rad_overlay,
                    'radial_mask_state': radial_mask_state, 'radial_label_state': radial_label_state,
                    'rad_tiff': rad_tiff, 'rad_lbl_tiff': rad_lbl_tiff,
                    'radial_table': radial_table, 'radial_csv': radial_csv,
                    'radial_tar_overlay': radial_tar_overlay, 'radial_ref_overlay': radial_ref_overlay,
                    'radial_tgt_on_and_img': radial_tgt_on_and_img, 'radial_ref_on_and_img': radial_ref_on_and_img,
                    'radial_ratio_img': radial_ratio_img, 'radial_quant_df_state': radial_quant_df_state,
                    'prof_start': prof_start, 'prof_end': prof_end, 'prof_window_size': prof_window_size,
                    'prof_window_step': prof_window_step, 'prof_smoothing': prof_smoothing, 'prof_show_err': prof_show_err,
                    'profile_show_ratio': profile_show_ratio,
                    'run_prof_btn': run_prof_btn, 'profile_table_raw': profile_table_raw, 'profile_csv': profile_csv, 'profile_table_sg': profile_table_sg, 'profile_csv_sg': profile_csv_sg,
                    'run_prof_single_btn': run_prof_single_btn, 'profile_plot': profile_plot,
                    'peak_min_pct': peak_min_pct, 'peak_max_pct': peak_max_pct,
                    'peak_algo': peak_algo,
                    'sg_window': sg_window, 'sg_poly': sg_poly,
                    'peak_slope_eps_rel': peak_slope_eps_rel,
                    'run_peak_diff_btn': run_peak_diff_btn, 'peak_diff_table': peak_diff_table,
                    'peak_diff_csv': peak_diff_csv, 'peak_diff_state': peak_diff_state,
                    'label_scale': label_scale,
                }
                create_stepbystep_callbacks(stepbystep_components)

                # ---------------- Persist settings (Step-by-Step tab) ----------------
                demo.load(
                    fn=None,
                    inputs=[],
                    outputs=[
                        seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu, seg_drop_edge,
                        pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method,
                        rad_in, rad_out, rad_min_obj,
                        # Target mask: channel, mode, pct, sat_limit, min_obj (standardized order)
                        tgt_chan, tgt_mask_mode, tgt_pct, tgt_sat_limit, tgt_min_obj,
                        # Reference mask: channel, mode, pct, sat_limit, min_obj
                        ref_chan, ref_mask_mode, ref_pct, ref_sat_limit, ref_min_obj,
                        px_w, px_h,
                        prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, profile_show_ratio,
                        peak_min_pct, peak_max_pct, peak_algo, sg_window, sg_poly, peak_slope_eps_rel,
                        ratio_eps,
                        label_scale,
                    ],
                    js=f"""
                    () => {{
                        try {{
                            const raw = localStorage.getItem('{SETTINGS_KEY}');
                            const d = {{
                                seg_source: 'target', seg_chan: 'gray', diameter: 0, flow_th: 0.4, cellprob_th: 0.0, use_gpu: true, seg_drop_edge: true,
                                pp_bg_enable: false, pp_bg_mode: 'rolling', pp_bg_radius: 50, pp_dark_pct: 5.0, pp_norm_enable: false, pp_norm_method: 'z-score',
                                rad_in: 0.0, rad_out: 100.0, rad_min_obj: 50,
                                tgt_chan: 'gray', tgt_mask_mode: 'none', tgt_sat_limit: 254, tgt_pct: 75.0, tgt_min_obj: 50,
                                ref_chan: 'gray', ref_mask_mode: 'none', ref_sat_limit: 254, ref_pct: 75.0, ref_min_obj: 50,
                                px_w: 1.0, px_h: 1.0,
                                prof_start: 0.0, prof_end: 150.0, prof_window_size: 10.0, prof_window_step: 5.0, prof_smoothing: 1, prof_show_err: true, profile_show_ratio: true,
                                peak_min_pct: 60.0, peak_max_pct: 120.0, peak_algo: 'global_max', sg_window: 5, sg_poly: 2, peak_slope_eps_rel: 0.001,
                                ratio_eps: 1e-6,
                                label_scale: {float(LABEL_SCALE)},
                            }};
                            let s = raw ? {{...d, ...JSON.parse(raw)}} : d;
                            const mapChan = (v) => ({{0:'gray',1:'R',2:'G',3:'B'}})[v] ?? v;
                            s.seg_chan = mapChan(s.seg_chan);
                            s.tgt_chan = mapChan(s.tgt_chan);
                            s.ref_chan = mapChan(s.ref_chan);
                            // sanitize deprecated algo
                            if (s.peak_algo === 'first_shoulder') s.peak_algo = 'first_local_top';
                            return [
                                s.seg_source,
                                s.seg_chan,
                                s.diameter,
                                s.flow_th,
                                s.cellprob_th,
                                s.use_gpu,
                                s.seg_drop_edge,
                                s.pp_bg_enable,
                                s.pp_bg_mode,
                                s.pp_bg_radius,
                                s.pp_dark_pct,
                                s.pp_norm_enable,
                                s.pp_norm_method,
                                s.rad_in,
                                s.rad_out,
                                s.rad_min_obj,
                                // Target mask standardized ordering
                                s.tgt_chan,
                                s.tgt_mask_mode,
                                s.tgt_pct,
                                s.tgt_sat_limit,
                                s.tgt_min_obj,
                                // Reference mask standardized ordering
                                s.ref_chan,
                                s.ref_mask_mode,
                                s.ref_pct,
                                s.ref_sat_limit,
                                s.ref_min_obj,
                                s.px_w,
                                s.px_h,
                                s.prof_start,
                                s.prof_end,
                                s.prof_window_size,
                                s.prof_window_step,
                                s.prof_smoothing,
                                s.prof_show_err,
                                s.profile_show_ratio,
                                s.peak_min_pct,
                                s.peak_max_pct,
                                s.peak_algo,
                                s.sg_window,
                                s.sg_poly,
                                s.peak_slope_eps_rel,
                                s.ratio_eps,
                                s.label_scale,
                            ];
                        }} catch (e) {{
                            console.warn('Failed to load saved settings:', e);
                            return [
                                'target', 'gray', 0, 0.4, 0.0, true, true,
                                false, 'rolling', 50, 5.0, false, 'z-score',
                                0.0, 100.0, 50,
                                'gray', 'global_percentile', 75.0, 254, 50,
                                'gray', 'global_percentile', 75.0, 254, 50,
                                1.0, 1.0,
                                0.0, 150.0, 10.0, 5.0, 1, true, true,
                                60.0, 120.0, 'global_max', 5, 2, 0.001,
                                1e-6,
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
                    (seg_source, 'seg_source'), (seg_chan, 'seg_chan'), (diameter, 'diameter'), (flow_th, 'flow_th'), (cellprob_th, 'cellprob_th'), (use_gpu, 'use_gpu'), (seg_drop_edge, 'seg_drop_edge'),
                    (pp_bg_enable, 'pp_bg_enable'), (pp_bg_mode, 'pp_bg_mode'), (pp_bg_radius, 'pp_bg_radius'), (pp_dark_pct, 'pp_dark_pct'), (pp_norm_enable, 'pp_norm_enable'), (pp_norm_method, 'pp_norm_method'),
                    (rad_in, 'rad_in'), (rad_out, 'rad_out'), (rad_min_obj, 'rad_min_obj'),
                    (tgt_chan, 'tgt_chan'), (tgt_mask_mode, 'tgt_mask_mode'), (tgt_sat_limit, 'tgt_sat_limit'), (tgt_pct, 'tgt_pct'), (tgt_min_obj, 'tgt_min_obj'),
                    (ref_chan, 'ref_chan'), (ref_mask_mode, 'ref_mask_mode'), (ref_sat_limit, 'ref_sat_limit'), (ref_pct, 'ref_pct'), (ref_min_obj, 'ref_min_obj'),
                    (px_w, 'px_w'), (px_h, 'px_h'), (label_scale, 'label_scale'),
                    (prof_start, 'prof_start'), (prof_end, 'prof_end'), (prof_window_size, 'prof_window_size'), (prof_window_step, 'prof_window_step'), (prof_smoothing, 'prof_smoothing'), (prof_show_err, 'prof_show_err'), (profile_show_ratio, 'profile_show_ratio'), (peak_min_pct, 'peak_min_pct'), (peak_max_pct, 'peak_max_pct'), (peak_algo, 'peak_algo'), (sg_window, 'sg_window'), (sg_poly, 'sg_poly'), (peak_slope_eps_rel, 'peak_slope_eps_rel'), (ratio_eps, 'ratio_eps'),
                ]:
                    _persist_change(comp, key)

                def _set_label_scale(v: float):
                    # Update package-level LABEL_SCALE so visualization uses the latest value
                    try:
                        dcq.LABEL_SCALE = float(v)
                    except Exception:
                        pass
                    return None
                label_scale.change(fn=_set_label_scale, inputs=[label_scale], outputs=[])

                # Initialize package-level value from saved settings on page load
                demo.load(
                    fn=_set_label_scale,
                    inputs=[label_scale_init_state],
                    outputs=[],
                    js=f"""
                    () => {{
                        try {{
                            const raw = localStorage.getItem('{SETTINGS_KEY}');
                            if (raw) {{
                                const s = JSON.parse(raw);
                                if (s && typeof s.label_scale !== 'undefined') return s.label_scale;
                            }}
                        }} catch (e) {{}}
                        return {float(LABEL_SCALE)};
                    }}
                    """,
                )

                reset_settings.click(
                    fn=None,
                    inputs=[],
                    outputs=[
                        seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu, seg_drop_edge,
                        pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method,
                        rad_in, rad_out, rad_min_obj,
                        tgt_chan, tgt_mask_mode, tgt_pct, tgt_sat_limit, tgt_min_obj,
                        ref_chan, ref_mask_mode, ref_pct, ref_sat_limit, ref_min_obj,
                        px_w, px_h,
                        prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, profile_show_ratio,
                        peak_min_pct, peak_max_pct, peak_algo, sg_window, sg_poly, peak_slope_eps_rel, ratio_eps,
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
                            'target', 'gray', 0, 0.4, 0.0, true, true,
                            false, 'dark_subtract', 50, 5.0, false, 'min-max',
                            0.0, 100.0, 50,
                            'gray', 'none', 75.0, 254, 50,
                            'gray', 'none', 75.0, 254, 50,
                            1.0, 1.0,
                            0.0, 150.0, 10.0, 5.0, 1, true, true,
                            60.0, 120.0, 'global_max', 5, 2, 0.001, {float(1e-6)},
                            {float(LABEL_SCALE)},
                        ];
                    }}
                    """,
                )
            
            # ==================== Tab 2: Radial Profile (Steps 1→2→3→5→6) ====================
            with gr.TabItem("⚡ Radial Profile", id=1) as tab_radialprofile:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Input Images")
                        with gr.Row():
                            tgt_quick = gr.Image(type="pil", label="Target image", image_mode="RGB", width=600)
                            ref_quick = gr.Image(type="pil", label="Reference image", image_mode="RGB", width=600)
                        
                        gr.Markdown("## Settings")
                        gr.Markdown("*⚙️ Settings are shared with Step-by-Step tab via browser storage*")
                        
                        with gr.Accordion("1. Segmentation settings", open=False):
                            seg_source_q = gr.Radio(["target", "reference"], value="target", label="Segment on")
                            seg_chan_q = gr.Radio(["gray", "R", "G", "B"], value="gray", label="Channel")
                            diameter_q = gr.Slider(0, 200, value=0, step=1, label="Diameter (0=auto)")
                            flow_th_q = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                            cellprob_th_q = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")
                            use_gpu_q = gr.Checkbox(value=True, label="Use GPU")
                            seg_drop_edge_q = gr.Checkbox(value=True, label="Exclude edge-touching labels")
                        
                        with gr.Accordion("2. Mask settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**Target**")
                                    tgt_chan_q = gr.Radio(["gray", "R", "G", "B"], value="gray", label="Ch")
                                    tgt_mask_mode_q = gr.Dropdown(
                                        ["none", "global_percentile", "global_otsu", "per_cell_percentile", "per_cell_otsu"],
                                        value="none", label="Mode")
                                    tgt_pct_q = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Pct")
                                    tgt_sat_limit_q = gr.Slider(0, 255, value=254, step=1, label="Sat")
                                    tgt_min_obj_q = gr.Slider(0, 2000, value=50, step=10, label="Min")
                                with gr.Column():
                                    gr.Markdown("**Reference**")
                                    ref_chan_q = gr.Radio(["gray", "R", "G", "B"], value="gray", label="Ch")
                                    ref_mask_mode_q = gr.Dropdown(
                                        ["none", "global_percentile", "global_otsu", "per_cell_percentile", "per_cell_otsu"],
                                        value="none", label="Mode")
                                    ref_pct_q = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Pct")
                                    ref_sat_limit_q = gr.Slider(0, 255, value=254, step=1, label="Sat")
                                    ref_min_obj_q = gr.Slider(0, 2000, value=50, step=10, label="Min")

                        with gr.Accordion("3. Preprocessing settings", open=False):
                            with gr.Column():
                                pp_bg_enable_q = gr.Checkbox(value=False, label="Background correction")
                                pp_bg_mode_q = gr.Dropdown(["rolling", "dark_subtract", "manual"], 
                                                        value="dark_subtract", label="BG method")
                                pp_bg_radius_q = gr.Slider(1, 300, value=50, step=1, label="Rolling radius")
                                pp_dark_pct_q = gr.Slider(0.0, 50.0, value=5.0, step=0.5, label="Dark pct")
                                with gr.Row():
                                    bak_tar_q = gr.Number(value=1.0, label="Target BG")
                                    bak_ref_q = gr.Number(value=1.0, label="Ref BG")
                            with gr.Column():
                                pp_norm_enable_q = gr.Checkbox(value=False, label="Normalization")
                                pp_norm_method_q = gr.Dropdown(
                                    ["z-score", "robust z-score", "min-max", "percentile [1,99]"],
                                    value="min-max", label="Method")
                            with gr.Column():
                                ratio_eps_q = gr.Number(value=1e-6, label="Ratio epsilon")
                                with gr.Row():
                                    px_w_q = gr.Number(value=1.0, label="Pixel width (µm)")
                                    px_h_q = gr.Number(value=1.0, label="Pixel height (µm)")
                        
                        with gr.Accordion("5. Radial profile settings", open=True):
                            prof_start_q = gr.Number(value=0.0, label="Start %")
                            prof_end_q = gr.Number(value=150.0, label="End %")
                            prof_window_size_q = gr.Number(value=10.0, label="Window size %")
                            prof_window_step_q = gr.Number(value=5.0, label="Window step %")
                            # Deprecated (moving average). Hidden; SG is now used for plotting/detection.
                            prof_smoothing_q = gr.Number(value=1, label="[deprecated] Smoothing (moving avg)", visible=False)

                        

                        
                        
                        gr.Markdown("---")
                        gr.Markdown("## 🚀 Run Analysis")
                        run_full_btn = gr.Button(
                            "Run Full Pipeline (Steps 1→2→3→5)", 
                            variant="primary", 
                            size="lg"
                        )
                        progress_text = gr.Textbox(label="Progress", interactive=False, lines=6)
                    
                        gr.Markdown("## 📊 Results")
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    integrate_tar_overlay = gr.Image(type="pil", label="Target overlay (AND mask)", width=600)
                                    integrate_ref_overlay = gr.Image(type="pil", label="Reference overlay (AND mask)", width=600)
                                # quant_table_q = gr.Dataframe(label="Per-cell quantification", 
                                #                             interactive=False, pinned_columns=1)
                                # quant_csv_q = gr.File(label="Download quantification CSV")
                                quant_table_q=gr.State()
                                quant_csv_q=gr.State()
                        
                        with gr.Tabs():

                            with gr.TabItem("📈 Profile Plot"):
                                
                                with gr.Row():
                                    sg_window_q = gr.Number(value=5, label="SG window (odd)", precision=0)
                                    sg_poly_q = gr.Number(value=2, label="SG poly", precision=0)
                                prof_show_err_q = gr.Checkbox(value=True, label="Show error bars")
                                profile_show_ratio_q = gr.Checkbox(value=True, label="Show T/R ratio curve")
                                run_prof_single_btn_q = gr.Button("Redraw (no peaks)")
                                profile_plot_q = gr.Image(type="pil", label="Radial intensity profile (3-col grid)", width=900)
                            
                            with gr.TabItem("📋 Profile Data"):
                                with gr.Tabs():
                                    with gr.TabItem("RAW"):
                                        profile_table_q_raw = gr.Dataframe(label="Radial profile (RAW)", interactive=False, pinned_columns=1)
                                        profile_csv_q = gr.File(label="Download RAW profile CSV")
                                    with gr.TabItem("SG (smoothed)"):
                                        profile_table_q_sg = gr.Dataframe(label="Radial profile (SG-applied)", interactive=False, pinned_columns=1)
                                        profile_csv_q_sg = gr.File(label="Download SG profile CSV")
                        
                        gr.Markdown("## ⚡ Peak Difference Analysis")
                        with gr.Accordion("6. Peak analysis settings", open=True):
                            with gr.Row():    
                                sg_window_q = gr.Number(value=5, label="SG window (odd)", precision=0)
                                sg_poly_q = gr.Number(value=2, label="SG poly", precision=0)
                            with gr.Row():
                                peak_min_pct_q = gr.Number(value=60.0, label="Min %")
                                peak_max_pct_q = gr.Number(value=120.0, label="Max %")
                                peak_algo_q = gr.Dropdown([
                                    "global_max",
                                    "first_local_top",
                                ], value="global_max", label="Peak algorithm")
                                peak_slope_eps_rel_q = gr.Number(value=0.001, label="Slope eps (relative)")
                            


                            run_peak_diff_btn_q = gr.Button("Compute Peak Differences")
                            peak_diff_table_q = gr.Dataframe(label="Peak differences", 
                                                            interactive=False, pinned_columns=1)
                            peak_diff_csv_q = gr.File(label="Download peak CSV")
                
                # State variables for caching (Radial Profile tab)
                prof_cache_df_state_q = gr.State()
                prof_cache_csv_state_q = gr.State()
                prof_cache_plot_state_q = gr.State()
                prof_cache_params_state_q = gr.State()
                peak_diff_state_q = gr.State()
                
                # ==================== Wire up Radial Profile callbacks ====================
                radialprofile_components = {
                    'tgt_quick': tgt_quick, 'ref_quick': ref_quick,
                    'seg_source_q': seg_source_q, 'seg_chan_q': seg_chan_q, 'diameter_q': diameter_q,
                    'flow_th_q': flow_th_q, 'cellprob_th_q': cellprob_th_q, 'use_gpu_q': use_gpu_q, 'seg_drop_edge_q': seg_drop_edge_q,
                    'tgt_chan_q': tgt_chan_q, 'tgt_mask_mode_q': tgt_mask_mode_q, 'tgt_pct_q': tgt_pct_q,
                    'tgt_sat_limit_q': tgt_sat_limit_q, 'tgt_min_obj_q': tgt_min_obj_q,
                    'ref_chan_q': ref_chan_q, 'ref_mask_mode_q': ref_mask_mode_q, 'ref_pct_q': ref_pct_q,
                    'ref_sat_limit_q': ref_sat_limit_q, 'ref_min_obj_q': ref_min_obj_q,
                    'pp_bg_enable_q': pp_bg_enable_q, 'pp_bg_mode_q': pp_bg_mode_q, 'pp_bg_radius_q': pp_bg_radius_q,
                    'pp_dark_pct_q': pp_dark_pct_q, 'bak_tar_q': bak_tar_q, 'bak_ref_q': bak_ref_q,
                    'pp_norm_enable_q': pp_norm_enable_q, 'pp_norm_method_q': pp_norm_method_q,
                    'prof_start_q': prof_start_q, 'prof_end_q': prof_end_q, 'prof_window_size_q': prof_window_size_q,
                    'prof_window_step_q': prof_window_step_q, 'prof_smoothing_q': prof_smoothing_q, 'prof_show_err_q': prof_show_err_q,
                    'peak_min_pct_q': peak_min_pct_q, 'peak_max_pct_q': peak_max_pct_q, 'peak_algo_q': peak_algo_q,
                    'sg_window_q': sg_window_q, 'sg_poly_q': sg_poly_q,
                    'peak_slope_eps_rel_q': peak_slope_eps_rel_q,
                    'ratio_eps_q': ratio_eps_q, 'px_w_q': px_w_q, 'px_h_q': px_h_q,
                    'run_full_btn': run_full_btn, 'progress_text': progress_text,
                    'integrate_tar_overlay': integrate_tar_overlay, 'integrate_ref_overlay': integrate_ref_overlay,
                    'quant_table_q': quant_table_q, 'quant_csv_q': quant_csv_q,
                    'run_prof_single_btn_q': run_prof_single_btn_q,
                    'profile_show_ratio_q': profile_show_ratio_q, 'profile_plot_q': profile_plot_q, 'profile_table_q_raw': profile_table_q_raw, 'profile_csv_q': profile_csv_q, 'profile_table_q_sg': profile_table_q_sg, 'profile_csv_q_sg': profile_csv_q_sg,
                    'run_peak_diff_btn_q': run_peak_diff_btn_q, 'peak_diff_table_q': peak_diff_table_q, 'peak_diff_csv_q': peak_diff_csv_q,
                    'masks_state': masks_state, 'quant_df_state': quant_df_state,
                    'prof_cache_df_state_q': prof_cache_df_state_q, 'prof_cache_csv_state_q': prof_cache_csv_state_q,
                    'prof_cache_plot_state_q': prof_cache_plot_state_q, 'prof_cache_params_state_q': prof_cache_params_state_q,
                    'peak_diff_state_q': peak_diff_state_q,
                }
                create_radialprofile_callbacks(radialprofile_components)
                
                # Load settings from localStorage on tab load (Radial Profile)
                demo.load(
                    fn=None,
                    inputs=[],
                    outputs=[
                        seg_source_q, seg_chan_q, diameter_q, flow_th_q, cellprob_th_q, use_gpu_q, seg_drop_edge_q,
                        tgt_chan_q, tgt_mask_mode_q, tgt_pct_q, tgt_sat_limit_q, tgt_min_obj_q,
                        ref_chan_q, ref_mask_mode_q, ref_pct_q, ref_sat_limit_q, ref_min_obj_q,
                        pp_bg_enable_q, pp_bg_mode_q, pp_bg_radius_q, pp_dark_pct_q,
                        pp_norm_enable_q, pp_norm_method_q,
                        prof_start_q, prof_end_q, prof_window_size_q, prof_window_step_q,
                        prof_smoothing_q, prof_show_err_q, profile_show_ratio_q,
                        peak_min_pct_q, peak_max_pct_q, peak_algo_q, sg_window_q, sg_poly_q, peak_slope_eps_rel_q,
                        ratio_eps_q, px_w_q, px_h_q,
                    ],
                    js=f"""
                    () => {{
                        try {{
                            const raw = localStorage.getItem('{SETTINGS_KEY}');
                            const d = {{
                                seg_source: 'target', seg_chan: 'gray', diameter: 0, flow_th: 0.4, cellprob_th: 0.0, use_gpu: true, seg_drop_edge: true,
                                tgt_chan: 'gray', tgt_mask_mode: 'none', tgt_pct: 75.0, tgt_sat_limit: 254, tgt_min_obj: 50,
                                ref_chan: 'gray', ref_mask_mode: 'none', ref_pct: 75.0, ref_sat_limit: 254, ref_min_obj: 50,
                                pp_bg_enable: false, pp_bg_mode: 'dark_subtract', pp_bg_radius: 50, pp_dark_pct: 5.0,
                                pp_norm_enable: false, pp_norm_method: 'min-max',
                                prof_start: 0.0, prof_end: 150.0, prof_window_size: 10.0, prof_window_step: 5.0,
                                prof_smoothing: 1, prof_show_err: true, profile_show_ratio: true,
                                peak_min_pct: 60.0, peak_max_pct: 120.0, peak_algo: 'global_max', sg_window: 5, sg_poly: 2, peak_slope_eps_rel: 0.001,
                                ratio_eps: 1e-6, px_w: 1.0, px_h: 1.0,
                            }};
                            let s = raw ? {{...d, ...JSON.parse(raw)}} : d;
                            const mapChan = (v) => ({{0:'gray',1:'R',2:'G',3:'B'}})[v] ?? v;
                            s.seg_chan = mapChan(s.seg_chan);
                            s.tgt_chan = mapChan(s.tgt_chan);
                            s.ref_chan = mapChan(s.ref_chan);
                            // sanitize deprecated algo
                            if (s.peak_algo === 'first_shoulder') s.peak_algo = 'first_local_top';
                            return [
                                s.seg_source, s.seg_chan, s.diameter, s.flow_th, s.cellprob_th, s.use_gpu, s.seg_drop_edge,
                                s.tgt_chan, s.tgt_mask_mode, s.tgt_pct, s.tgt_sat_limit, s.tgt_min_obj,
                                s.ref_chan, s.ref_mask_mode, s.ref_pct, s.ref_sat_limit, s.ref_min_obj,
                                s.pp_bg_enable, s.pp_bg_mode, s.pp_bg_radius, s.pp_dark_pct,
                                s.pp_norm_enable, s.pp_norm_method,
                                s.prof_start, s.prof_end, s.prof_window_size, s.prof_window_step,
                                s.prof_smoothing, s.prof_show_err, s.profile_show_ratio,
                                s.peak_min_pct, s.peak_max_pct, s.peak_algo,
                                s.sg_window, s.sg_poly, s.peak_slope_eps_rel,
                                s.ratio_eps, s.px_w, s.px_h,
                            ];
                        }} catch (e) {{
                            console.warn('Failed to load saved settings:', e);
                            return Array(36).fill(null);
                        }}
                    }}
                    """,
                )
                
                # Save settings to localStorage on change (Radial Profile tab)
                def _persist_change_q(comp, key):
                    """Auto-save setting to localStorage when changed."""
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
                    (seg_source_q, 'seg_source'), (seg_chan_q, 'seg_chan'), (diameter_q, 'diameter'), 
                    (flow_th_q, 'flow_th'), (cellprob_th_q, 'cellprob_th'), (use_gpu_q, 'use_gpu'), (seg_drop_edge_q, 'seg_drop_edge'),
                    (tgt_chan_q, 'tgt_chan'), (tgt_mask_mode_q, 'tgt_mask_mode'), (tgt_pct_q, 'tgt_pct'),
                    (tgt_sat_limit_q, 'tgt_sat_limit'), (tgt_min_obj_q, 'tgt_min_obj'),
                    (ref_chan_q, 'ref_chan'), (ref_mask_mode_q, 'ref_mask_mode'), (ref_pct_q, 'ref_pct'),
                    (ref_sat_limit_q, 'ref_sat_limit'), (ref_min_obj_q, 'ref_min_obj'),
                    (pp_bg_enable_q, 'pp_bg_enable'), (pp_bg_mode_q, 'pp_bg_mode'), 
                    (pp_bg_radius_q, 'pp_bg_radius'), (pp_dark_pct_q, 'pp_dark_pct'),
                    (pp_norm_enable_q, 'pp_norm_enable'), (pp_norm_method_q, 'pp_norm_method'),
                    (prof_start_q, 'prof_start'), (prof_end_q, 'prof_end'), 
                    (prof_window_size_q, 'prof_window_size'), (prof_window_step_q, 'prof_window_step'),
                    (prof_smoothing_q, 'prof_smoothing'), (prof_show_err_q, 'prof_show_err'), (profile_show_ratio_q, 'profile_show_ratio'),
                    (peak_min_pct_q, 'peak_min_pct'), (peak_max_pct_q, 'peak_max_pct'), (peak_algo_q, 'peak_algo'), (sg_window_q, 'sg_window'), (sg_poly_q, 'sg_poly'), (peak_slope_eps_rel_q, 'peak_slope_eps_rel'),
                    (ratio_eps_q, 'ratio_eps'), (px_w_q, 'px_w'), (px_h_q, 'px_h'),
                ]:
                    _persist_change_q(comp, key)
        
        # ==================== Tab Switch Events: Reload Settings ====================
        # When switching to Step-by-Step tab, reload settings from localStorage
        tab_stepbystep.select(
            fn=None,
            inputs=[],
            outputs=[
                seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu, seg_drop_edge,
                tgt_chan, tgt_mask_mode, tgt_pct, tgt_sat_limit, tgt_min_obj,
                ref_chan, ref_mask_mode, ref_pct, ref_sat_limit, ref_min_obj,
                pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct,
                pp_norm_enable, pp_norm_method,
                prof_start, prof_end, prof_window_size, prof_window_step,
                prof_smoothing, prof_show_err, profile_show_ratio,
                peak_min_pct, peak_max_pct, peak_algo, sg_window, sg_poly,
                ratio_eps, px_w, px_h,
            ],
            js=f"""
            () => {{
                try {{
                    const raw = localStorage.getItem('{SETTINGS_KEY}');
                    const d = {{
                        seg_source: 'target', seg_chan: 'gray', diameter: 0, flow_th: 0.4, cellprob_th: 0.0, use_gpu: true,
                        tgt_chan: 'gray', tgt_mask_mode: 'none', tgt_pct: 75.0, tgt_sat_limit: 254, tgt_min_obj: 50,
                        ref_chan: 'gray', ref_mask_mode: 'none', ref_pct: 75.0, ref_sat_limit: 254, ref_min_obj: 50,
                        pp_bg_enable: false, pp_bg_mode: 'dark_subtract', pp_bg_radius: 50, pp_dark_pct: 5.0,
                        pp_norm_enable: false, pp_norm_method: 'min-max',
                        prof_start: 0.0, prof_end: 150.0, prof_window_size: 10.0, prof_window_step: 5.0,
                                prof_smoothing: 1, prof_show_err: true, profile_show_ratio: true,
                        peak_min_pct: 60.0, peak_max_pct: 120.0, peak_algo: 'global_max', sg_window: 5, sg_poly: 2,
                        ratio_eps: 1e-6, px_w: 1.0, px_h: 1.0,
                    }};
                    let s = raw ? {{...d, ...JSON.parse(raw)}} : d;
                    const mapChan = (v) => ({{0:'gray',1:'R',2:'G',3:'B'}})[v] ?? v;
                    s.seg_chan = mapChan(s.seg_chan);
                    s.tgt_chan = mapChan(s.tgt_chan);
                    s.ref_chan = mapChan(s.ref_chan);
                    // sanitize deprecated algo
                    if (s.peak_algo === 'first_shoulder') s.peak_algo = 'first_local_top';
                    return [
                        s.seg_source, s.seg_chan, s.diameter, s.flow_th, s.cellprob_th, s.use_gpu, s.seg_drop_edge,
                        s.tgt_chan, s.tgt_mask_mode, s.tgt_pct, s.tgt_sat_limit, s.tgt_min_obj,
                        s.ref_chan, s.ref_mask_mode, s.ref_pct, s.ref_sat_limit, s.ref_min_obj,
                        s.pp_bg_enable, s.pp_bg_mode, s.pp_bg_radius, s.pp_dark_pct,
                        s.pp_norm_enable, s.pp_norm_method,
                        s.prof_start, s.prof_end, s.prof_window_size, s.prof_window_step,
                        s.prof_smoothing, s.prof_show_err, s.profile_show_ratio,
                        s.peak_min_pct, s.peak_max_pct, s.peak_algo, s.sg_window, s.sg_poly,
                        s.ratio_eps, s.px_w, s.px_h,
                    ];
                }} catch (e) {{
                    console.warn('Failed to reload settings on tab switch:', e);
                    return Array(37).fill(null);
                }}
            }}
            """,
        )
        
        # When switching to Radial Profile tab, reload settings from localStorage
        tab_radialprofile.select(
            fn=None,
            inputs=[],
            outputs=[
                seg_source_q, seg_chan_q, diameter_q, flow_th_q, cellprob_th_q, use_gpu_q, seg_drop_edge_q,
                tgt_chan_q, tgt_mask_mode_q, tgt_pct_q, tgt_sat_limit_q, tgt_min_obj_q,
                ref_chan_q, ref_mask_mode_q, ref_pct_q, ref_sat_limit_q, ref_min_obj_q,
                pp_bg_enable_q, pp_bg_mode_q, pp_bg_radius_q, pp_dark_pct_q,
                pp_norm_enable_q, pp_norm_method_q,
                prof_start_q, prof_end_q, prof_window_size_q, prof_window_step_q,
                prof_smoothing_q, prof_show_err_q, profile_show_ratio_q,
                peak_min_pct_q, peak_max_pct_q, peak_algo_q, sg_window_q, sg_poly_q,
                ratio_eps_q, px_w_q, px_h_q,
            ],
            js=f"""
            () => {{
                try {{
                    const raw = localStorage.getItem('{SETTINGS_KEY}');
                    const d = {{
                        seg_source: 'target', seg_chan: 'gray', diameter: 0, flow_th: 0.4, cellprob_th: 0.0, use_gpu: true,
                        tgt_chan: 'gray', tgt_mask_mode: 'none', tgt_pct: 75.0, tgt_sat_limit: 254, tgt_min_obj: 50,
                        ref_chan: 'gray', ref_mask_mode: 'none', ref_pct: 75.0, ref_sat_limit: 254, ref_min_obj: 50,
                        pp_bg_enable: false, pp_bg_mode: 'dark_subtract', pp_bg_radius: 50, pp_dark_pct: 5.0,
                        pp_norm_enable: false, pp_norm_method: 'min-max',
                        prof_start: 0.0, prof_end: 150.0, prof_window_size: 10.0, prof_window_step: 5.0,
                        prof_smoothing: 1, prof_show_err: true, profile_show_ratio: true,
                        peak_min_pct: 60.0, peak_max_pct: 120.0, peak_algo: 'global_max', sg_window: 5, sg_poly: 2,
                        ratio_eps: 1e-6, px_w: 1.0, px_h: 1.0,
                    }};
                    let s = raw ? {{...d, ...JSON.parse(raw)}} : d;
                    const mapChan = (v) => ({{0:'gray',1:'R',2:'G',3:'B'}})[v] ?? v;
                    s.seg_chan = mapChan(s.seg_chan);
                    s.tgt_chan = mapChan(s.tgt_chan);
                    s.ref_chan = mapChan(s.ref_chan);
                    return [
                        s.seg_source, s.seg_chan, s.diameter, s.flow_th, s.cellprob_th, s.use_gpu, s.seg_drop_edge,
                        s.tgt_chan, s.tgt_mask_mode, s.tgt_pct, s.tgt_sat_limit, s.tgt_min_obj,
                        s.ref_chan, s.ref_mask_mode, s.ref_pct, s.ref_sat_limit, s.ref_min_obj,
                        s.pp_bg_enable, s.pp_bg_mode, s.pp_bg_radius, s.pp_dark_pct,
                        s.pp_norm_enable, s.pp_norm_method,
                        s.prof_start, s.prof_end, s.prof_window_size, s.prof_window_step,
                        s.prof_smoothing, s.prof_show_err, s.profile_show_ratio,
                        s.peak_min_pct, s.peak_max_pct, s.peak_algo, s.sg_window, s.sg_poly,
                        s.ratio_eps, s.px_w, s.px_h,
                    ];
                }} catch (e) {{
                    console.warn('Failed to reload settings on tab switch:', e);
                    return Array(37).fill(null);
                }}
            }}
            """,
        )
    
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch()
