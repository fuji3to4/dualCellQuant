"""
Callback functions for Radial Profile tab.
This file can be MODIFIED for improvements.
"""

import gradio as gr
import tempfile
import traceback
import pandas as pd
import numpy as np

from dualcellquant import *


def create_radialprofile_callbacks(components):
    """
    Wire up all callbacks for Radial Profile tab.
    
    Args:
        components: Dict containing all Gradio components from the UI
    """
    
    # Extract components
    tgt_quick = components['tgt_quick']
    ref_quick = components['ref_quick']
    
    seg_source_q = components['seg_source_q']
    seg_chan_q = components['seg_chan_q']
    diameter_q = components['diameter_q']
    flow_th_q = components['flow_th_q']
    cellprob_th_q = components['cellprob_th_q']
    use_gpu_q = components['use_gpu_q']
    
    tgt_chan_q = components['tgt_chan_q']
    tgt_mask_mode_q = components['tgt_mask_mode_q']
    tgt_pct_q = components['tgt_pct_q']
    tgt_sat_limit_q = components['tgt_sat_limit_q']
    tgt_min_obj_q = components['tgt_min_obj_q']
    
    ref_chan_q = components['ref_chan_q']
    ref_mask_mode_q = components['ref_mask_mode_q']
    ref_pct_q = components['ref_pct_q']
    ref_sat_limit_q = components['ref_sat_limit_q']
    ref_min_obj_q = components['ref_min_obj_q']
    
    pp_bg_enable_q = components['pp_bg_enable_q']
    pp_bg_mode_q = components['pp_bg_mode_q']
    pp_bg_radius_q = components['pp_bg_radius_q']
    pp_dark_pct_q = components['pp_dark_pct_q']
    bak_tar_q = components['bak_tar_q']
    bak_ref_q = components['bak_ref_q']
    pp_norm_enable_q = components['pp_norm_enable_q']
    pp_norm_method_q = components['pp_norm_method_q']
    
    prof_start_q = components['prof_start_q']
    prof_end_q = components['prof_end_q']
    prof_window_size_q = components['prof_window_size_q']
    prof_window_step_q = components['prof_window_step_q']
    prof_smoothing_q = components['prof_smoothing_q']
    prof_show_err_q = components['prof_show_err_q']
    profile_show_ratio_q = components.get('profile_show_ratio_q')
    
    peak_min_pct_q = components['peak_min_pct_q']
    peak_max_pct_q = components['peak_max_pct_q']
    peak_algo_q = components.get('peak_algo_q')
    
    ratio_eps_q = components['ratio_eps_q']
    px_w_q = components['px_w_q']
    px_h_q = components['px_h_q']
    
    run_full_btn = components['run_full_btn']
    progress_text = components['progress_text']
    
    integrate_tar_overlay = components['integrate_tar_overlay']
    integrate_ref_overlay = components['integrate_ref_overlay']
    quant_table_q = components['quant_table_q']
    quant_csv_q = components['quant_csv_q']
    
    run_prof_single_btn_q = components['run_prof_single_btn_q']
    profile_plot_q = components['profile_plot_q']
    profile_table_q_raw = components['profile_table_q_raw']
    profile_table_q_sg = components['profile_table_q_sg']
    profile_csv_q = components['profile_csv_q']
    profile_csv_q_sg = components['profile_csv_q_sg']
    
    run_peak_diff_btn_q = components['run_peak_diff_btn_q']
    peak_diff_table_q = components['peak_diff_table_q']
    peak_diff_csv_q = components['peak_diff_csv_q']
    
    masks_state = components['masks_state']
    quant_df_state = components['quant_df_state']
    
    # Cache state variables
    prof_cache_df_state_q = components['prof_cache_df_state_q']
    prof_cache_csv_state_q = components['prof_cache_csv_state_q']
    prof_cache_plot_state_q = components['prof_cache_plot_state_q']
    prof_cache_params_state_q = components['prof_cache_params_state_q']
    peak_diff_state_q = components['peak_diff_state_q']
    
    # ==================== Callback Functions ====================
    
    # Full pipeline callback
    def _run_full_pipeline(tgt_img, ref_img, seg_src, seg_ch, diam, flow, cellprob, gpu, drop_edge,
                          t_ch, t_mode, t_pct, t_sat, t_min,
                          r_ch, r_mode, r_pct, r_sat, r_min,
                          bg_en, bg_mode, bg_rad, dark_pct, bg_t, bg_r, nm_en, nm_m,
                          p_start, p_end, p_win, p_step, p_smooth, p_err, p_show_ratio,
                          eps, pw, ph, sg_w, sg_p):
        """Run steps 1→2→3→5→6 in sequence."""
        if tgt_img is None or ref_img is None:
            return (
                "❌ Please upload both target and reference images.",
                None, None, None, None,  # profile tables RAW/SG, csvs (RAW/SG), plot
                None, None,         # quant table, csv
                None, None,         # integrate overlays
                None, None,         # bak_tar_q, bak_ref_q
                None, None,         # masks_state, quant_df_state
                None, None, None,   # prof_cache_df_state, csv_state, plot_state
                None,               # prof_cache_params_state
                None                # peak_diff_state
            )
        
        progress = []
        
        try:
            # Step 1: Segmentation
            progress.append("🔄 Step 1/5: Running Cellpose segmentation...")
            yield (
                "\n".join(progress),
                None, None, None, None, None,  # profile: RAW tbl, RAW csv, SG tbl, SG csv, plot
                None, None,         # quant table, csv
                None, None,         # integrate overlays
                None, None,         # bak_tar_q, bak_ref_q
                None, None,         # masks_state, quant_df_state
                None, None, None, None,  # prof_cache df,csv,plot,params
                None                # peak_diff_state
            )
            
            _, _, _, masks = run_segmentation(
                tgt_img, ref_img, seg_src, seg_ch, diam, flow, cellprob, gpu,
                drop_edge_cells=bool(drop_edge), inside_fraction_min=0.98, edge_margin_pct=1.0,
            )
            progress[-1] = "✅ Step 1/5: Segmentation complete"
            yield (
                "\n".join(progress),
                None, None, None, None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None, None,
                None
            )
            
            # Count labels (no dropdown in this tab)
            try:
                labs = np.unique(masks)
                labs = labs[labs > 0]
                lab_count = len(labs)
            except Exception:
                lab_count = 0
            
            # Step 2: Apply Masks
            progress.append("🔄 Step 2/5: Applying target and reference masks...")
            yield (
                "\n".join(progress),
                None, None, None, None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None, None,
                None
            )
            
            _, _, tgt_mask = apply_mask(
                tgt_img, masks, t_ch, t_sat, t_mode, t_pct, t_min, None, "target_mask"
            )
            _, _, ref_mask = apply_mask(
                ref_img, masks, r_ch, r_sat, r_mode, r_pct, r_min, None, "reference_mask"
            )
            progress[-1] = "✅ Step 2/5: Masks applied"
            yield (
                "\n".join(progress),
                None, None, None, None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None, None,
                None
            )
            
            # Step 3: Integrate & Quantify (without radial mask - step 4 is skipped)
            progress.append("🔄 Step 3/5: Integrating and quantifying...")
            yield (
                "\n".join(progress),
                None, None, None, None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None, None,
                None
            )
            
            bgm = str(bg_mode)
            # UX parity with Step-by-Step: compute display BG values when dark_subtract
            out_tar_bg = bg_t
            out_ref_bg = bg_r
            if bool(bg_en) and bgm == "dark_subtract":
                try:
                    out_tar_bg = compute_dark_background(tgt_img, t_ch, float(dark_pct), use_native_scale=True)
                except Exception:
                    out_tar_bg = bg_t
                try:
                    out_ref_bg = compute_dark_background(ref_img, r_ch, float(dark_pct), use_native_scale=True)
                except Exception:
                    out_ref_bg = bg_r
            mt = float(bg_t) if (bg_en and bgm == "manual") else None
            mr = float(bg_r) if (bg_en and bgm == "manual") else None
            
            integrate_tar_ov, integrate_ref_ov, _, quant_df, quant_csv, _, _, _ = integrate_and_quantify(
                tgt_img, ref_img, masks, tgt_mask, ref_mask,
                t_ch, r_ch, pw, ph,
                bg_en, bg_rad, nm_en, nm_m,
                bg_mode=bgm, bg_dark_pct=dark_pct,
                manual_tar_bg=mt, manual_ref_bg=mr,
                roi_mask=None, roi_labels=None,  # No radial mask
                ratio_ref_epsilon=eps
            )
            progress[-1] = "✅ Step 3/5: Quantification complete"
            yield (
                "\n".join(progress),
                None, None, None, None, None,
                quant_df, quant_csv,
                integrate_tar_ov, integrate_ref_ov,
                out_tar_bg, out_ref_bg,
                None, None,
                None, None, None, None,
                None
            )
            
            # Step 5: Radial Profile
            progress.append("🔄 Step 4/5: Computing radial intensity profiles...")
            yield (
                "\n".join(progress),
                None, None, None, None, None,
                quant_df, quant_csv,
                integrate_tar_ov, integrate_ref_ov,
                out_tar_bg, out_ref_bg,
                None, None,
                None, None, None, None,
                None
            )
            
            df_all, csv_all = radial_profile_all_cells(
                tgt_img, ref_img, masks,
                t_ch, r_ch,
                p_start, p_end, p_win, p_step,
                bg_en, bg_rad, nm_en, nm_m,
                bg_mode=bgm, bg_dark_pct=dark_pct,
                manual_tar_bg=mt, manual_ref_bg=mr,
                ratio_ref_epsilon=eps
            )
            
            _, _, plot_img = radial_profile_analysis(
                tgt_img, ref_img, masks,
                t_ch, r_ch,
                p_start, p_end, p_win, p_step,
                bg_en, bg_rad, nm_en, nm_m,
                bg_mode=bgm, bg_dark_pct=dark_pct,
                manual_tar_bg=mt, manual_ref_bg=mr,
                window_bins=int(p_smooth),
                show_errorbars=bool(p_err),
                ratio_ref_epsilon=eps
            )
            
            # Build cache params
            try:
                labs = np.unique(masks)
                labs = labs[labs > 0]
                lab_count = len(labs)
                lab_max = int(np.max(labs)) if len(labs) > 0 else 0
                mshape = masks.shape if masks is not None else None
            except Exception:
                lab_count = 0; lab_max = 0; mshape = None
            
            cache_params = dict(
                tchan=str(t_ch), rchan=str(r_ch), start=float(p_start), end=float(p_end),
                window_size=float(p_win), window_step=float(p_step),
                bg_enable=bool(bg_en), bg_mode=str(bg_mode), bg_radius=int(bg_rad), dark_pct=float(dark_pct),
                norm_enable=bool(nm_en), norm_method=str(nm_m),
                man_t=mt, man_r=mr,
                ratio_eps=float(eps),
                mask_shape=mshape, lab_count=lab_count, lab_max=lab_max,
            )
            
            # Build a 3-column grid image for UI display (use show_ratio setting)
            try:
                labs = np.unique(masks)
                labs = labs[labs > 0]
                label_list = [int(l) for l in labs]
            except Exception:
                label_list = []
            # Initial grid should be RAW (no SG); force sg_window=1
            grid_img = build_radial_profile_grid_image(
                df_all, peak_df=None, window_bins=int(p_smooth), sg_window=1, sg_poly=int(sg_p), show_errorbars=bool(p_err), show_ratio=bool(p_show_ratio), labels=label_list, cols=3, tile_width=700
            )
            progress[-1] = "✅ Step 4/5: Radial profile analysis complete"
            yield (
                "\n".join(progress),
                df_all, csv_all, None, None, grid_img,
                quant_df, quant_csv,
                integrate_tar_ov, integrate_ref_ov,
                out_tar_bg, out_ref_bg,
                None, None,
                df_all, csv_all, grid_img, cache_params,
                None
            )
            
            # Step 6 is manual (peak analysis button)
            progress.append(f"✅ Step 5/5: Analysis complete! Found {lab_count} cells.")
            progress.append("\n💡 Tip: Use 'Peak Analysis' tab to compute peak differences")
            
            yield (
                "\n".join(progress),
                df_all, csv_all, None, None, grid_img,
                quant_df, quant_csv,
                integrate_tar_ov, integrate_ref_ov,
                out_tar_bg, out_ref_bg,
                masks, quant_df,
                df_all, csv_all, grid_img, cache_params,
                None
            )
            
        except Exception as e:
            error_msg = f"❌ Error during analysis:\n{str(e)}\n\n{traceback.format_exc()}"
            progress.append(error_msg)
            yield (
                "\n".join(progress),
                None, None, None, None, None,
                None, None,
                None, None,
                None, None,
                None, None,
                None, None, None, None,
                None
            )
    
    run_full_btn.click(
        fn=_run_full_pipeline,
        inputs=[
            tgt_quick, ref_quick, seg_source_q, seg_chan_q, diameter_q, flow_th_q, cellprob_th_q, use_gpu_q, components['seg_drop_edge_q'],
            tgt_chan_q, tgt_mask_mode_q, tgt_pct_q, tgt_sat_limit_q, tgt_min_obj_q,
            ref_chan_q, ref_mask_mode_q, ref_pct_q, ref_sat_limit_q, ref_min_obj_q,
            pp_bg_enable_q, pp_bg_mode_q, pp_bg_radius_q, pp_dark_pct_q, bak_tar_q, bak_ref_q,
            pp_norm_enable_q, pp_norm_method_q,
            prof_start_q, prof_end_q, prof_window_size_q, prof_window_step_q, prof_smoothing_q, prof_show_err_q, profile_show_ratio_q,
            ratio_eps_q, px_w_q, px_h_q,
            components['sg_window_q'], components['sg_poly_q']
        ],
        outputs=[
            progress_text,
            profile_table_q_raw, profile_csv_q, profile_table_q_sg, profile_csv_q_sg, profile_plot_q,
            quant_table_q, quant_csv_q,
            integrate_tar_overlay, integrate_ref_overlay,
            bak_tar_q, bak_ref_q,
            masks_state, quant_df_state,
            prof_cache_df_state_q, prof_cache_csv_state_q, prof_cache_plot_state_q, prof_cache_params_state_q, peak_diff_state_q
        ],
    )
    
    # Single cell profile update (with caching like Step-by-Step)
    def _update_single_profile(tgt_img, ref_img, masks, 
                              t_ch, r_ch,
                              p_start, p_end, p_win, p_step, p_smooth, p_err, show_ratio,
                              bg_en, bg_mode, bg_rad, dark_pct, nm_en, nm_m,
                              bg_t, bg_r, eps,
                              sg_w, sg_p,
                              cache_df, cache_csv, cache_plot, cache_params, peak_df):
        """Redraw profile grid (apply SG), and generate SG-smoothed table. Uses cache when possible."""
        # Always ignore any previously computed peaks for a pure redraw
        peak_df = None
        if masks is None:
            return None, None, None, cache_df, cache_csv, cache_plot, cache_params
        
        bgm = str(bg_mode)
        mt = float(bg_t) if (bg_en and bgm == "manual") else None
        mr = float(bg_r) if (bg_en and bgm == "manual") else None
        
        # Build current params
        try:
            labs = np.unique(masks)
            labs = labs[labs > 0]
            lab_count = len(labs)
            lab_max = int(np.max(labs)) if len(labs) > 0 else 0
            mshape = masks.shape if masks is not None else None
        except Exception:
            lab_count = 0; lab_max = 0; mshape = None
        
        cur_params = dict(
            tchan=str(t_ch), rchan=str(r_ch), start=float(p_start), end=float(p_end),
            window_size=float(p_win), window_step=float(p_step),
            bg_enable=bool(bg_en), bg_mode=str(bg_mode), bg_radius=int(bg_rad), dark_pct=float(dark_pct),
            norm_enable=bool(nm_en), norm_method=str(nm_m),
            man_t=mt, man_r=mr,
            ratio_eps=float(eps),
            mask_shape=mshape, lab_count=lab_count, lab_max=lab_max,
        )
        
        def params_equal(a, b):
            try:
                return a == b
            except Exception:
                return False
        
        # Always rebuild grid for all labels
        if (cache_df is not None) and params_equal(cache_params, cur_params):
            try:
                labs_now = np.unique(masks)
                labs_now = labs_now[labs_now > 0]
                label_list_now = [int(l) for l in labs_now]
            except Exception:
                label_list_now = []
            title_suffix = "(SG)" if int(sg_w) >= 3 else "(RAW)"
            grid_img = build_radial_profile_grid_image(
                cache_df, peak_df,
                window_bins=int(p_smooth), sg_window=int(sg_w), sg_poly=int(sg_p), show_errorbars=bool(p_err), show_ratio=bool(show_ratio),
                labels=label_list_now, cols=3, tile_width=700, title_suffix=title_suffix
            )
            # Build SG-smoothed table from cache
            df_sg = _apply_sg_to_profile_df(cache_df, int(sg_w), int(sg_p))
            tmp_csv_sg = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_sg.csv"); df_sg.to_csv(tmp_csv_sg.name, index=False)
            return cache_df, cache_csv, df_sg, tmp_csv_sg.name, grid_img, cache_df, cache_csv, grid_img, cache_params

        # Recompute all
        df_all, csv_all = radial_profile_all_cells(
            tgt_img, ref_img, masks, t_ch, r_ch,
            float(p_start), float(p_end), float(p_win), float(p_step),
            bool(bg_en), int(bg_rad), bool(nm_en), nm_m,
            bg_mode=bgm, bg_dark_pct=float(dark_pct),
            manual_tar_bg=mt, manual_ref_bg=mr, ratio_ref_epsilon=float(eps),
        )
        try:
            labs_now = np.unique(masks)
            labs_now = labs_now[labs_now > 0]
            label_list_now = [int(l) for l in labs_now]
        except Exception:
            label_list_now = []
        title_suffix = "(SG)" if int(sg_w) >= 3 else "(RAW)"
        grid_img = build_radial_profile_grid_image(
            df_all, peak_df,
            window_bins=int(p_smooth), sg_window=int(sg_w), sg_poly=int(sg_p), show_errorbars=bool(p_err), show_ratio=bool(show_ratio),
            labels=label_list_now, cols=3, tile_width=700, title_suffix=title_suffix
        )
        # Build SG-smoothed table
        df_sg = _apply_sg_to_profile_df(df_all, int(sg_w), int(sg_p))
        tmp_csv_sg = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_sg.csv"); df_sg.to_csv(tmp_csv_sg.name, index=False)
        return df_all, csv_all, df_sg, tmp_csv_sg.name, grid_img, df_all, csv_all, grid_img, cur_params
    
    run_prof_single_btn_q.click(
        fn=_update_single_profile,
        inputs=[
            tgt_quick, ref_quick, masks_state,
            tgt_chan_q, ref_chan_q,
            prof_start_q, prof_end_q, prof_window_size_q, prof_window_step_q,
            prof_smoothing_q, prof_show_err_q, profile_show_ratio_q,
            pp_bg_enable_q, pp_bg_mode_q, pp_bg_radius_q, pp_dark_pct_q,
            pp_norm_enable_q, pp_norm_method_q,
            bak_tar_q, bak_ref_q, ratio_eps_q,
            components['sg_window_q'], components['sg_poly_q'],
            prof_cache_df_state_q, prof_cache_csv_state_q, prof_cache_plot_state_q, prof_cache_params_state_q, peak_diff_state_q
        ],
        outputs=[profile_table_q_raw, profile_csv_q, profile_table_q_sg, profile_csv_q_sg, profile_plot_q, prof_cache_df_state_q, prof_cache_csv_state_q, prof_cache_plot_state_q, prof_cache_params_state_q],
    )
    
    # Peak difference callback
    def _compute_peaks(cache_df, quant_df, min_pct, max_pct, smoothing, show_err, show_ratio, peak_algo, sg_window, sg_poly, peak_slope_eps_rel):
        """Compute peak differences and update plot as a 3-col grid for all labels, honoring T/R toggle."""
        if cache_df is None or cache_df.empty:
            return gr.update(value=pd.DataFrame()), None, gr.update(), None
        
        algo = str(peak_algo) if peak_algo is not None else 'global_max'
        try:
            w = int(sg_window) if sg_window is not None else 5
        except Exception:
            w = 5
        try:
            p = int(sg_poly) if sg_poly is not None else 2
        except Exception:
            p = 2
        try:
            slope_rel = float(peak_slope_eps_rel) if peak_slope_eps_rel is not None else 0.001
        except Exception:
            slope_rel = 0.001
        peak_df = compute_radial_peak_difference(cache_df, quant_df, float(min_pct), float(max_pct), algo=algo, sg_window=w, sg_poly=p, peak_slope_eps_rel=slope_rel)
        # Ensure intensity columns are present and column order is user-friendly
        try:
            preferred_cols = [
                "label",
                "max_target_center_pct", "max_reference_center_pct", "difference_pct",
                "max_target_px", "max_reference_px", "difference_px",
                "max_target_um", "max_reference_um", "difference_um",
                "max_target_intensity", "max_reference_intensity", "ratio_intensity",
                # Reference RAW quality metrics
                "ref_range_rel", "ref_noise_rel", "ref_neg_run_after_peak", "accept_ref",
            ]
            cols = [c for c in preferred_cols if c in peak_df.columns] + [c for c in peak_df.columns if c not in preferred_cols]
            peak_df = peak_df[cols]
        except Exception:
            pass
        
        if peak_df.empty:
            return gr.update(value=pd.DataFrame()), None, gr.update(), None
        
        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_peak_difference.csv")
        peak_df.to_csv(tmp_csv.name, index=False)
        tmp_csv_path = tmp_csv.name
        tmp_csv.close()
        
        # Build grid image with peak markers
        try:
            # Always render all labels
            labels = sorted(int(x) for x in cache_df["label"].dropna().unique())
            plot_img = build_radial_profile_grid_image(
                cache_df, peak_df,
                window_bins=int(smoothing), sg_window=int(w), sg_poly=int(p), show_errorbars=bool(show_err), show_ratio=bool(show_ratio),
                labels=labels, cols=3, tile_width=700, title_suffix="(with peaks)"
            )
        except Exception:
            plot_img = None
        
        return peak_df, tmp_csv_path, plot_img, peak_df
    
    run_peak_diff_btn_q.click(
        fn=_compute_peaks,
        inputs=[prof_cache_df_state_q, quant_df_state, peak_min_pct_q, peak_max_pct_q, prof_smoothing_q, prof_show_err_q, profile_show_ratio_q, peak_algo_q, components['sg_window_q'], components['sg_poly_q'], components['peak_slope_eps_rel_q']],
        outputs=[peak_diff_table_q, peak_diff_csv_q, profile_plot_q, peak_diff_state_q],
    )

    # Helper: Apply SG smoothing per label to means for Quick tab as well
    def _apply_sg_to_profile_df(df: pd.DataFrame, sg_w: int, sg_p: int) -> pd.DataFrame:
        try:
            if df is None or df.empty:
                return df
            if sg_w is None or int(sg_w) < 3:
                return df.copy()
            def _smooth_group(g: pd.DataFrame) -> pd.DataFrame:
                g = g.sort_values('center_pct').copy()
                from scipy.signal import savgol_filter as _sg
                def _smooth_arr(a: np.ndarray) -> np.ndarray:
                    a = a.astype(float)
                    n = a.size
                    if n < 3 or int(sg_w) < 3:
                        return a
                    w = int(sg_w)
                    if w > n:
                        w = n if (n % 2 == 1) else n - 1
                    if w % 2 == 0:
                        w -= 1
                    p = min(int(sg_p), max(1, w - 1))
                    if w < 3:
                        return a
                    # NaN-safe via simple interpolation
                    m = np.isfinite(a)
                    if m.sum() >= 2:
                        af = a.copy(); af[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), a[m])
                    else:
                        af = a
                    try:
                        return _sg(af, window_length=w, polyorder=p, mode='interp')
                    except Exception:
                        return a
                g['mean_target'] = _smooth_arr(g['mean_target'].to_numpy(dtype=float))
                g['mean_reference'] = _smooth_arr(g['mean_reference'].to_numpy(dtype=float))
                if 'mean_ratio_T_over_R' in g.columns:
                    g['mean_ratio_T_over_R'] = _smooth_arr(g['mean_ratio_T_over_R'].to_numpy(dtype=float))
                return g
            # Avoid FutureWarning from pandas GroupBy.apply including grouping columns
            parts = []
            for _, g in df.groupby('label', sort=False):
                parts.append(_smooth_group(g))
            out = pd.concat(parts, ignore_index=True) if parts else df.copy()
            return out
        except Exception:
            return df.copy()

    # Toggle BG control visibility based on mode (parity with Step-by-Step)
    def _pp_bg_mode_changed_q(mode: str):
        m = (mode or "rolling").lower()
        return (
            gr.update(visible=(m == "rolling")),
            gr.update(visible=(m == "dark_subtract")),
            gr.update(visible=(m in ("manual", "dark_subtract"))),
            gr.update(visible=(m in ("manual", "dark_subtract"))),
        )

    pp_bg_mode_q.change(
        fn=_pp_bg_mode_changed_q,
        inputs=[pp_bg_mode_q],
        outputs=[pp_bg_radius_q, pp_dark_pct_q, bak_tar_q, bak_ref_q],
    )
