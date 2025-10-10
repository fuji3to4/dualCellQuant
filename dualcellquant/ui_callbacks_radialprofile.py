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
    
    peak_min_pct_q = components['peak_min_pct_q']
    peak_max_pct_q = components['peak_max_pct_q']
    
    ratio_eps_q = components['ratio_eps_q']
    px_w_q = components['px_w_q']
    px_h_q = components['px_h_q']
    
    run_full_btn = components['run_full_btn']
    progress_text = components['progress_text']
    
    integrate_tar_overlay = components['integrate_tar_overlay']
    integrate_ref_overlay = components['integrate_ref_overlay']
    quant_table_q = components['quant_table_q']
    quant_csv_q = components['quant_csv_q']
    
    prof_label_q = components['prof_label_q']
    run_prof_single_btn_q = components['run_prof_single_btn_q']
    profile_plot_q = components['profile_plot_q']
    profile_table_q = components['profile_table_q']
    profile_csv_q = components['profile_csv_q']
    
    run_peak_diff_btn_q = components['run_peak_diff_btn_q']
    peak_diff_table_q = components['peak_diff_table_q']
    peak_diff_csv_q = components['peak_diff_csv_q']
    
    masks_state = components['masks_state']
    quant_df_state = components['quant_df_state']
    
    # ==================== Callback Functions ====================
    
    # Full pipeline callback
    def _run_full_pipeline(tgt_img, ref_img, seg_src, seg_ch, diam, flow, cellprob, gpu,
                          t_ch, t_mode, t_pct, t_sat, t_min,
                          r_ch, r_mode, r_pct, r_sat, r_min,
                          bg_en, bg_mode, bg_rad, dark_pct, bg_t, bg_r, nm_en, nm_m,
                          p_start, p_end, p_win, p_step, p_smooth, p_err,
                          eps, pw, ph):
        """Run steps 1→2→3→5→6 in sequence."""
        if tgt_img is None or ref_img is None:
            return ("❌ Please upload both target and reference images.", 
                    None, None, None, None, None, None, None, None, None, gr.update())
        
        progress = []
        
        try:
            # Step 1: Segmentation
            progress.append("🔄 Step 1/5: Running Cellpose segmentation...")
            yield ("\n".join(progress), None, None, None, None, None, None, None, None, None, gr.update())
            
            _, _, _, masks = run_segmentation(
                tgt_img, ref_img, seg_src, seg_ch, diam, flow, cellprob, gpu
            )
            progress[-1] = "✅ Step 1/5: Segmentation complete"
            yield ("\n".join(progress), None, None, None, None, None, None, None, None, None, gr.update())
            
            # Build label choices
            try:
                labs = np.unique(masks)
                labs = labs[labs > 0]
                choices = ["All"] + [str(int(l)) for l in labs]
                lab_count = len(labs)
            except Exception:
                choices = ["All"]
                lab_count = 0
            
            # Step 2: Apply Masks
            progress.append("🔄 Step 2/5: Applying target and reference masks...")
            yield ("\n".join(progress), None, None, None, None, None, None, None, None, None, gr.update())
            
            _, _, tgt_mask = apply_mask(
                tgt_img, masks, t_ch, t_sat, t_mode, t_pct, t_min, None, "target_mask"
            )
            _, _, ref_mask = apply_mask(
                ref_img, masks, r_ch, r_sat, r_mode, r_pct, r_min, None, "reference_mask"
            )
            progress[-1] = "✅ Step 2/5: Masks applied"
            yield ("\n".join(progress), None, None, None, None, None, None, None, None, None, gr.update())
            
            # Step 3: Integrate & Quantify (without radial mask - step 4 is skipped)
            progress.append("🔄 Step 3/5: Integrating and quantifying...")
            yield ("\n".join(progress), None, None, None, None, None, None, None, None, None, gr.update())
            
            bgm = str(bg_mode)
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
            yield ("\n".join(progress), None, None, None, quant_df, quant_csv, integrate_tar_ov, integrate_ref_ov, None, None, gr.update())
            
            # Step 5: Radial Profile
            progress.append("🔄 Step 4/5: Computing radial intensity profiles...")
            yield ("\n".join(progress), None, None, None, quant_df, quant_csv, integrate_tar_ov, integrate_ref_ov, None, None, gr.update())
            
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
            
            progress[-1] = "✅ Step 4/5: Radial profile analysis complete"
            yield ("\n".join(progress), df_all, csv_all, plot_img, quant_df, quant_csv, integrate_tar_ov, integrate_ref_ov, None, None, gr.update())
            
            # Step 6 is manual (peak analysis button)
            progress.append(f"✅ Step 5/5: Analysis complete! Found {lab_count} cells.")
            progress.append("\n💡 Tip: Use 'Peak Analysis' tab to compute peak differences")
            
            yield (
                "\n".join(progress),
                df_all, csv_all, plot_img,
                quant_df, quant_csv,
                integrate_tar_ov, integrate_ref_ov,
                masks, quant_df,
                gr.update(choices=choices, value="All")
            )
            
        except Exception as e:
            error_msg = f"❌ Error during analysis:\n{str(e)}\n\n{traceback.format_exc()}"
            progress.append(error_msg)
            yield ("\n".join(progress), None, None, None, None, None, None, None, None, None, gr.update())
    
    run_full_btn.click(
        fn=_run_full_pipeline,
        inputs=[
            tgt_quick, ref_quick, seg_source_q, seg_chan_q, diameter_q, flow_th_q, cellprob_th_q, use_gpu_q,
            tgt_chan_q, tgt_mask_mode_q, tgt_pct_q, tgt_sat_limit_q, tgt_min_obj_q,
            ref_chan_q, ref_mask_mode_q, ref_pct_q, ref_sat_limit_q, ref_min_obj_q,
            pp_bg_enable_q, pp_bg_mode_q, pp_bg_radius_q, pp_dark_pct_q, bak_tar_q, bak_ref_q,
            pp_norm_enable_q, pp_norm_method_q,
            prof_start_q, prof_end_q, prof_window_size_q, prof_window_step_q, prof_smoothing_q, prof_show_err_q,
            ratio_eps_q, px_w_q, px_h_q
        ],
        outputs=[
            progress_text,
            profile_table_q, profile_csv_q, profile_plot_q,
            quant_table_q, quant_csv_q,
            integrate_tar_overlay, integrate_ref_overlay,
            masks_state, quant_df_state,
            prof_label_q
        ],
    )
    
    # Single cell profile update
    def _update_single_profile(tgt_img, ref_img, masks, label_val, t_ch, r_ch,
                              p_start, p_end, p_win, p_step, p_smooth, p_err,
                              bg_en, bg_mode, bg_rad, dark_pct, nm_en, nm_m,
                              bg_t, bg_r, eps, cache_df):
        if masks is None or cache_df is None:
            return None, None, None
        
        bgm = str(bg_mode)
        mt = float(bg_t) if (bg_en and bgm == "manual") else None
        mr = float(bg_r) if (bg_en and bgm == "manual") else None
        
        if str(label_val) == "All":
            # Rebuild plot from cache
            _, _, plot_img = radial_profile_analysis(
                tgt_img, ref_img, masks, t_ch, r_ch,
                p_start, p_end, p_win, p_step,
                bg_en, bg_rad, nm_en, nm_m,
                bg_mode=bgm, bg_dark_pct=dark_pct,
                manual_tar_bg=mt, manual_ref_bg=mr,
                window_bins=int(p_smooth),
                show_errorbars=bool(p_err),
                ratio_ref_epsilon=eps
            )
            return cache_df, None, plot_img
        else:
            try:
                lab = int(str(label_val))
                df1, csv1, plot1 = radial_profile_single(
                    tgt_img, ref_img, masks, lab, t_ch, r_ch,
                    p_start, p_end, p_win, p_step,
                    bg_en, bg_rad, nm_en, nm_m,
                    bg_mode=bgm, bg_dark_pct=dark_pct,
                    manual_tar_bg=mt, manual_ref_bg=mr,
                    window_bins=int(p_smooth),
                    show_errorbars=bool(p_err),
                    ratio_ref_epsilon=eps
                )
                return df1, csv1, plot1
            except Exception:
                return None, None, None
    
    run_prof_single_btn_q.click(
        fn=_update_single_profile,
        inputs=[
            tgt_quick, ref_quick, masks_state, prof_label_q,
            tgt_chan_q, ref_chan_q,
            prof_start_q, prof_end_q, prof_window_size_q, prof_window_step_q,
            prof_smoothing_q, prof_show_err_q,
            pp_bg_enable_q, pp_bg_mode_q, pp_bg_radius_q, pp_dark_pct_q,
            pp_norm_enable_q, pp_norm_method_q,
            bak_tar_q, bak_ref_q, ratio_eps_q,
            profile_table_q
        ],
        outputs=[profile_table_q, profile_csv_q, profile_plot_q],
    )
    
    # Peak difference callback
    def _compute_peaks(cache_df, quant_df, min_pct, max_pct):
        if cache_df is None or cache_df.empty:
            return None, None
        
        peak_df = compute_radial_peak_difference(cache_df, quant_df, min_pct, max_pct)
        
        if peak_df.empty:
            return None, None
        
        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_peak_difference.csv")
        peak_df.to_csv(tmp_csv.name, index=False)
        tmp_csv.close()
        
        return peak_df, tmp_csv.name
    
    run_peak_diff_btn_q.click(
        fn=_compute_peaks,
        inputs=[profile_table_q, quant_df_state, peak_min_pct_q, peak_max_pct_q],
        outputs=[peak_diff_table_q, peak_diff_csv_q],
    )
