"""
Callback functions for Step-by-Step tab.
This file is STABLE - do not modify unless necessary.
"""

import gradio as gr
import tempfile
import pandas as pd
import numpy as np

from dualcellquant import *


def create_stepbystep_callbacks(components):
    """
    Wire up all callbacks for Step-by-Step tab.
    
    Args:
        components: Dict containing all Gradio components from the UI
    """
    
    # Extract components
    tgt = components['tgt']
    ref = components['ref']
    seg_source = components['seg_source']
    seg_chan = components['seg_chan']
    diameter = components['diameter']
    flow_th = components['flow_th']
    cellprob_th = components['cellprob_th']
    use_gpu = components['use_gpu']
    
    run_seg_btn = components['run_seg_btn']
    seg_overlay = components['seg_overlay']
    seg_tiff_file = components['seg_tiff_file']
    mask_img = components['mask_img']
    masks_state = components['masks_state']
    prof_label = components['prof_label']
    prof_cache_df_state = components['prof_cache_df_state']
    prof_cache_csv_state = components['prof_cache_csv_state']
    prof_cache_plot_state = components['prof_cache_plot_state']
    prof_cache_params_state = components['prof_cache_params_state']
    
    tgt_chan = components['tgt_chan']
    tgt_mask_mode = components['tgt_mask_mode']
    tgt_pct = components['tgt_pct']
    tgt_sat_limit = components['tgt_sat_limit']
    tgt_min_obj = components['tgt_min_obj']
    
    ref_chan = components['ref_chan']
    ref_mask_mode = components['ref_mask_mode']
    ref_pct = components['ref_pct']
    ref_sat_limit = components['ref_sat_limit']
    ref_min_obj = components['ref_min_obj']
    
    run_tgt_btn = components['run_tgt_btn']
    tgt_overlay = components['tgt_overlay']
    tgt_tiff = components['tgt_tiff']
    tgt_mask_state = components['tgt_mask_state']
    ref_overlay = components['ref_overlay']
    ref_tiff = components['ref_tiff']
    ref_mask_state = components['ref_mask_state']
    
    pp_bg_enable = components['pp_bg_enable']
    pp_bg_mode = components['pp_bg_mode']
    pp_bg_radius = components['pp_bg_radius']
    pp_dark_pct = components['pp_dark_pct']
    pp_norm_enable = components['pp_norm_enable']
    pp_norm_method = components['pp_norm_method']
    bak_tar = components['bak_tar']
    bak_ref = components['bak_ref']
    ratio_eps = components['ratio_eps']
    px_w = components['px_w']
    px_h = components['px_h']
    
    integrate_btn = components['integrate_btn']
    integrate_tar_overlay = components['integrate_tar_overlay']
    integrate_ref_overlay = components['integrate_ref_overlay']
    mask_tiff = components['mask_tiff']
    table = components['table']
    csv_file = components['csv_file']
    tgt_on_and_img = components['tgt_on_and_img']
    ref_on_and_img = components['ref_on_and_img']
    ratio_img = components['ratio_img']
    quant_df_state = components['quant_df_state']
    
    rad_in = components['rad_in']
    rad_out = components['rad_out']
    rad_min_obj = components['rad_min_obj']
    run_rad_btn = components['run_rad_btn']
    rad_overlay = components['rad_overlay']
    radial_mask_state = components['radial_mask_state']
    radial_label_state = components['radial_label_state']
    rad_tiff = components['rad_tiff']
    rad_lbl_tiff = components['rad_lbl_tiff']
    radial_table = components['radial_table']
    radial_csv = components['radial_csv']
    radial_tar_overlay = components['radial_tar_overlay']
    radial_ref_overlay = components['radial_ref_overlay']
    radial_tgt_on_and_img = components['radial_tgt_on_and_img']
    radial_ref_on_and_img = components['radial_ref_on_and_img']
    radial_ratio_img = components['radial_ratio_img']
    radial_quant_df_state = components['radial_quant_df_state']
    
    prof_start = components['prof_start']
    prof_end = components['prof_end']
    prof_window_size = components['prof_window_size']
    prof_window_step = components['prof_window_step']
    prof_smoothing = components['prof_smoothing']
    prof_show_err = components['prof_show_err']
    
    run_prof_btn = components['run_prof_btn']
    profile_table = components['profile_table']
    profile_csv = components['profile_csv']
    run_prof_single_btn = components['run_prof_single_btn']
    profile_plot = components['profile_plot']
    
    peak_min_pct = components['peak_min_pct']
    peak_max_pct = components['peak_max_pct']
    run_peak_diff_btn = components['run_peak_diff_btn']
    peak_diff_table = components['peak_diff_table']
    peak_diff_csv = components['peak_diff_csv']
    peak_diff_state = components['peak_diff_state']
    
    label_scale = components['label_scale']
    
    # ==================== Callback Functions ====================
    
    # Segmentation
    def _run_seg(tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu):
        ov, seg_tif, mask_viz, masks = run_segmentation(tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu)
        try:
            labs = np.unique(masks)
            labs = labs[labs > 0]
            choices = ["All"] + [str(int(l)) for l in labs]
        except Exception:
            choices = ["All"]
        return ov, seg_tif, mask_viz, masks, gr.update(choices=choices, value="All"), None, None, None, None
    
    run_seg_btn.click(
        fn=_run_seg,
        inputs=[tgt, ref, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu],
        outputs=[seg_overlay, seg_tiff_file, mask_img, masks_state, prof_label, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
    )
    
    # Radial mask
    def _radial_and_quantify(tgt_img, ref_img, masks, rin, rout, mino, tmask, rmask, tchan, rchan, pw, ph, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps):
        ov, rad_bool, rad_lbl, tiff_bool, tiff_lbl = radial_mask(masks, rin, rout, mino)
        bgm = str(bg_mode)
        mt = float(man_t) if (bg_en and bgm == "manual") else None
        mr = float(man_r) if (bg_en and bgm == "manual") else None
        q_tar_ov, q_ref_ov, q_and_tiff, q_df, q_csv, q_tgt_on, q_ref_on, q_ratio = integrate_and_quantify(
            tgt_img, ref_img, masks, tmask, rmask, tchan, rchan, pw, ph,
            bool(bg_en), int(bg_r), bool(nm_en), nm_m,
            bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
            manual_tar_bg=mt, manual_ref_bg=mr, roi_mask=rad_bool, roi_labels=rad_lbl, ratio_ref_epsilon=float(eps),
        )
        return ov, rad_bool, rad_lbl, tiff_bool, tiff_lbl, q_df, q_csv, q_tar_ov, q_ref_ov, q_tgt_on, q_ref_on, q_ratio, q_df
    
    run_rad_btn.click(
        fn=_radial_and_quantify,
        inputs=[tgt, ref, masks_state, rad_in, rad_out, rad_min_obj, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan, px_w, px_h, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps],
        outputs=[rad_overlay, radial_mask_state, radial_label_state, rad_tiff, rad_lbl_tiff, radial_table, radial_csv, radial_tar_overlay, radial_ref_overlay, radial_tgt_on_and_img, radial_ref_on_and_img, radial_ratio_img, radial_quant_df_state],
    )
    
    # Radial profile
    def _radial_profile_cb(tgt_img, ref_img, masks, tchan, rchan, s, e, wsize, wstep, smoothing, show_err, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps):
        bgm = str(bg_mode)
        mt = float(man_t) if (bg_en and bgm == "manual") else None
        mr = float(man_r) if (bg_en and bgm == "manual") else None
        df_all, csv_all = radial_profile_all_cells(
            tgt_img, ref_img, masks, tchan, rchan,
            float(s), float(e), float(wsize), float(wstep),
            bool(bg_en), int(bg_r), bool(nm_en), nm_m,
            bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
            manual_tar_bg=mt, manual_ref_bg=mr, ratio_ref_epsilon=float(eps),
        )
        _, _, plot_img = radial_profile_analysis(
            tgt_img, ref_img, masks, tchan, rchan,
            float(s), float(e), float(wsize), float(wstep),
            bool(bg_en), int(bg_r), bool(nm_en), nm_m,
            bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
            manual_tar_bg=mt, manual_ref_bg=mr,
            window_bins=int(smoothing), show_errorbars=bool(show_err), ratio_ref_epsilon=float(eps),
        )
        try:
            labs = np.unique(masks)
            labs = labs[labs > 0]
            lab_count = int(labs.size)
            lab_max = int(labs.max()) if lab_count > 0 else 0
            mshape = tuple(masks.shape)
        except Exception:
            lab_count = 0; lab_max = 0; mshape = None
        params = dict(
            tchan=str(tchan), rchan=str(rchan), start=float(s), end=float(e), 
            window_size=float(wsize), window_step=float(wstep),
            bg_enable=bool(bg_en), bg_mode=str(bg_mode), bg_radius=int(bg_r), dark_pct=float(dark_pct),
            norm_enable=bool(nm_en), norm_method=str(nm_m),
            man_t=float(mt) if mt is not None else None, man_r=float(mr) if mr is not None else None,
            ratio_eps=float(eps),
            mask_shape=mshape, lab_count=lab_count, lab_max=lab_max,
        )
        return df_all, csv_all, plot_img, df_all, csv_all, plot_img, params
    
    run_prof_btn.click(
        fn=_radial_profile_cb,
        inputs=[tgt, ref, masks_state, tgt_chan, ref_chan, prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps],
        outputs=[profile_table, profile_csv, profile_plot, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
    )
    
    def _radial_profile_single_or_all_cb(tgt_img, ref_img, masks, label_val, tchan, rchan, s, e, wsize, wstep, smoothing, show_err, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps, cache_df, cache_csv, cache_plot, cache_params, peak_df):
        bgm = str(bg_mode)
        mt = float(man_t) if (bg_en and bgm == "manual") else None
        mr = float(man_r) if (bg_en and bgm == "manual") else None
        try:
            labs_now = np.unique(masks)
            labs_now = labs_now[labs_now > 0]
            lab_count = int(labs_now.size)
            lab_max = int(labs_now.max()) if lab_count > 0 else 0
            mshape = tuple(masks.shape)
        except Exception:
            lab_count = 0; lab_max = 0; mshape = None
        cur_params = dict(
            tchan=str(tchan), rchan=str(rchan), start=float(s), end=float(e), 
            window_size=float(wsize), window_step=float(wstep),
            bg_enable=bool(bg_en), bg_mode=str(bg_mode), bg_radius=int(bg_r), dark_pct=float(dark_pct),
            norm_enable=bool(nm_en), norm_method=str(nm_m),
            man_t=float(mt) if mt is not None else None, man_r=float(mr) if mr is not None else None,
            ratio_eps=float(eps),
            mask_shape=mshape, lab_count=lab_count, lab_max=lab_max,
        )
        def params_equal(a, b):
            try:
                return a == b
            except Exception:
                return False
        if str(label_val) == "All":
            def _build_all_plot_from_df(df_all_in: pd.DataFrame, window_bins_int: int, show_err_bool: bool, peak_df_in: pd.DataFrame = None):
                return plot_radial_profile_with_peaks(df_all_in, peak_df_in, "All", window_bins_int, show_err_bool)

            if (cache_df is not None) and params_equal(cache_params, cur_params):
                plot_img = _build_all_plot_from_df(cache_df, int(smoothing), bool(show_err), peak_df)
                return cache_df, cache_csv, plot_img, cache_df, cache_csv, plot_img, cache_params
            df_all, csv_all = radial_profile_all_cells(
                tgt_img, ref_img, masks, tchan, rchan,
                float(s), float(e), float(wsize), float(wstep),
                bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                manual_tar_bg=mt, manual_ref_bg=mr, ratio_ref_epsilon=float(eps),
            )
            plot_img = _build_all_plot_from_df(df_all, int(smoothing), bool(show_err), peak_df)
            return df_all, csv_all, plot_img, df_all, csv_all, plot_img, cur_params
        else:
            try:
                lab = int(str(label_val))
            except Exception:
                lab = None
            if lab is None:
                return gr.update(), None, None, cache_df, cache_csv, cache_plot, cache_params
            use_df = None
            if (cache_df is not None) and params_equal(cache_params, cur_params):
                use_df = cache_df
            else:
                use_df, csv_all = radial_profile_all_cells(
                    tgt_img, ref_img, masks, tchan, rchan,
                    float(s), float(e), float(wsize), float(wstep),
                    bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                    bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                    manual_tar_bg=mt, manual_ref_bg=mr,
                )
                _, _, cache_plot_new = radial_profile_analysis(
                    tgt_img, ref_img, masks, tchan, rchan,
                    float(s), float(e), float(wsize), float(wstep),
                    bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                    bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                    manual_tar_bg=mt, manual_ref_bg=mr,
                    window_bins=int(smoothing), show_errorbars=bool(show_err), ratio_ref_epsilon=float(eps),
                )
                cache_df = use_df; cache_csv = csv_all; cache_plot = cache_plot_new; cache_params = cur_params
            try:
                df1 = use_df[use_df["label"] == int(lab)].copy()
            except Exception:
                df1, csv1, plot1 = radial_profile_single(
                    tgt_img, ref_img, masks, lab, tchan, rchan,
                    float(s), float(e), float(wsize), float(wstep),
                    bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                    bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                    manual_tar_bg=mt, manual_ref_bg=mr,
                    window_bins=int(smoothing), show_errorbars=bool(show_err), ratio_ref_epsilon=float(eps),
                )
                return df1, csv1, plot1, cache_df, cache_csv, cache_plot, cache_params
            tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=f"_radial_profile_label_{lab}.csv")
            df1.to_csv(tmp_csv.name, index=False)
            plot1 = plot_radial_profile_with_peaks(use_df, peak_df, lab, int(smoothing), bool(show_err))
            return df1, tmp_csv.name, plot1, cache_df, cache_csv, cache_plot, cache_params
    
    run_prof_single_btn.click(
        fn=_radial_profile_single_or_all_cb,
        inputs=[tgt, ref, masks_state, prof_label, tgt_chan, ref_chan, prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state, peak_diff_state],
        outputs=[profile_table, profile_csv, profile_plot, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
    )
    
    # Peak difference
    def _peak_diff_cb(cached_df, quant_df, min_pct, max_pct, label_val, smoothing, show_err):
        if cached_df is None or cached_df.empty:
            return gr.update(value=pd.DataFrame()), None, gr.update(), None
        
        peak_df = compute_radial_peak_difference(cached_df, quant_df, float(min_pct), float(max_pct))
        
        if peak_df.empty:
            return gr.update(value=pd.DataFrame()), None, gr.update(), None
        
        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_peak_difference.csv")
        peak_df.to_csv(tmp_csv.name, index=False)
        tmp_csv_path = tmp_csv.name
        tmp_csv.close()
        
        try:
            plot_img = plot_radial_profile_with_peaks(
                cached_df, peak_df, label_val, int(smoothing), bool(show_err), title_suffix="(with peaks)"
            )
        except Exception:
            plot_img = None
        
        return peak_df, tmp_csv_path, plot_img, peak_df
    
    run_peak_diff_btn.click(
        fn=_peak_diff_cb,
        inputs=[prof_cache_df_state, quant_df_state, peak_min_pct, peak_max_pct, prof_label, prof_smoothing, prof_show_err],
        outputs=[peak_diff_table, peak_diff_csv, profile_plot, peak_diff_state],
    )
    
    # Apply masks
    def _apply_masks_both(tgt_img, ref_img, m, t_ch, t_sat, t_mode, t_p, t_mino, r_ch, r_sat, r_mode, r_p, r_mino):
        t_ov, t_tiff_path, t_mask = apply_mask(tgt_img, m, t_ch, t_sat, t_mode, t_p, t_mino, None, "target_mask")
        r_ov, r_tiff_path, r_mask = apply_mask(ref_img, m, r_ch, r_sat, r_mode, r_p, r_mino, None, "reference_mask")
        return t_ov, t_tiff_path, t_mask, r_ov, r_tiff_path, r_mask
    
    run_tgt_btn.click(
        fn=_apply_masks_both,
        inputs=[tgt, ref, masks_state, tgt_chan, tgt_sat_limit, tgt_mask_mode, tgt_pct, tgt_min_obj, ref_chan, ref_sat_limit, ref_mask_mode, ref_pct, ref_min_obj],
        outputs=[tgt_overlay, tgt_tiff, tgt_mask_state, ref_overlay, ref_tiff, ref_mask_state],
    )

    # Toggle percentile slider visibility
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
    
    def _integrate_callback(tgt_img, ref_img, ms, tmask, rmask, tchan, rchan, pw, ph, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps):
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
        man_t = float(out_tar_bg) if (bg_en and bg_mode_s == "manual") else None
        man_r = float(out_ref_bg) if (bg_en and bg_mode_s == "manual") else None
        res = integrate_and_quantify(
            tgt_img, ref_img, ms, tmask, rmask, tchan, rchan,
            pw, ph,
            bool(bg_en), int(bg_r), bool(nm_en), nm_m,
            bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
            manual_tar_bg=man_t, manual_ref_bg=man_r, ratio_ref_epsilon=float(eps),
        )
        return (*res[:8], out_tar_bg, out_ref_bg, res[3])
    
    integrate_btn.click(
        fn=_integrate_callback,
        inputs=[tgt, ref, masks_state, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan, px_w, px_h, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps],
        outputs=[integrate_tar_overlay, integrate_ref_overlay, mask_tiff, table, csv_file, tgt_on_and_img, ref_on_and_img, ratio_img, bak_tar, bak_ref, quant_df_state],
    )

    def _set_label_scale(v: float):
        global LABEL_SCALE
        try:
            LABEL_SCALE = float(v)
        except Exception:
            pass
        return None
    
    label_scale.change(fn=_set_label_scale, inputs=[label_scale], outputs=[])
