"""
Callback functions for Step-by-Step tab.
This file is STABLE - do not modify unless necessary.
"""

import gradio as gr
import tempfile
"""
Callback functions for Step-by-Step tab.
This file is STABLE - do not modify unless necessary.
"""

import gradio as gr
import tempfile
import pandas as pd
import numpy as np

from dualcellquant import *
import dualcellquant as dcq


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
    seg_drop_edge = components['seg_drop_edge']
    
    run_seg_btn = components['run_seg_btn']
    seg_overlay = components['seg_overlay']
    seg_tiff_file = components['seg_tiff_file']
    mask_img = components['mask_img']
    masks_state = components['masks_state']
    # no label selector (always operate on All labels)
    prof_cache_df_state = components['prof_cache_df_state']
    prof_cache_csv_state = components['prof_cache_csv_state']
    prof_cache_plot_state = components['prof_cache_plot_state']
    prof_cache_params_state = components['prof_cache_params_state']
    
    # Relabel controls
    prev_label_tiff = components.get('prev_label_tiff')
    iou_match_th = components.get('iou_match_th')
    relabel_btn = components.get('relabel_btn')
    
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
    profile_show_ratio = components.get('profile_show_ratio')
    
    run_prof_btn = components['run_prof_btn']
    profile_table_raw = components['profile_table_raw']
    profile_table_sg = components['profile_table_sg']
    profile_csv = components['profile_csv']
    profile_csv_sg = components['profile_csv_sg']
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
    def _run_seg(tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu, drop_edge):
        # Align behavior with Quick tab: allow mild border contact but drop largely truncated cells
        ov, seg_tif, mask_viz, masks = run_segmentation(
            tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu,
            drop_edge_cells=bool(drop_edge), inside_fraction_min=0.98, edge_margin_pct=1.0,
        )
        # no label dropdown to update; just reset caches
        return ov, seg_tif, mask_viz, masks, None, None, None, None
    
    run_seg_btn.click(
        fn=_run_seg,
        inputs=[tgt, ref, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu, seg_drop_edge],
        outputs=[seg_overlay, seg_tiff_file, mask_img, masks_state, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
    )

    # Relabel current masks to previous IDs
    def _extract_path(obj):
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return obj.get('path') or obj.get('name')
        if isinstance(obj, (list, tuple)) and obj:
            it = obj[0]
            if isinstance(it, dict):
                return it.get('path') or it.get('name')
            if isinstance(it, str):
                return it
        return None

    def _relabel_cb(tgt_img, ref_img, seg_src, seg_ch, prev_file, curr_masks, iou_th):
        try:
            import PIL.Image as _PIL
            if curr_masks is None:
                return gr.update(), gr.update(), gr.update(), None, None, None, None, None
            pth = _extract_path(prev_file)
            if not pth:
                return gr.update(), gr.update(), gr.update(), None, None, None, None, None
            prev_arr = np.array(_PIL.open(pth))
            if prev_arr.ndim != 2:
                prev_arr = np.squeeze(prev_arr)
            prev_arr = prev_arr.astype(np.int32, copy=False)
            curr = np.array(curr_masks).astype(np.int32, copy=False)
            if prev_arr.shape != curr.shape:
                return gr.update(), gr.update(), gr.update(), None, None, None, None, None
            rel, df_map, _ = relabel_to_previous(prev_arr, curr, iou_threshold=float(iou_th))
            seg_arr = pil_to_numpy(tgt_img) if str(seg_src) == 'target' else pil_to_numpy(ref_img)
            seg_gray = extract_single_channel(seg_arr, seg_ch)
            ov = colorize_overlay(seg_gray, rel, None)
            ov = annotate_ids(ov, rel)
            mask_viz = vivid_label_image(rel)
            tiff_path = save_label_tiff(rel, "seg_labels_relabel")
            return ov, tiff_path, mask_viz, rel, None, None, None, None
        except Exception:
            return gr.update(), gr.update(), gr.update(), None, None, None, None, None

    if relabel_btn is not None:
        relabel_btn.click(
            fn=_relabel_cb,
            inputs=[tgt, ref, seg_source, seg_chan, prev_label_tiff, masks_state, iou_match_th],
            outputs=[seg_overlay, seg_tiff_file, mask_img, masks_state, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
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
    def _radial_profile_cb(tgt_img, ref_img, masks, tchan, rchan, s, e, wsize, wstep, smoothing, show_err, show_ratio, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps, sg_w, sg_p):
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
        # Build grid (3 columns, variable rows) of all single-cell plots
        try:
            labs = np.unique(masks)
            labs = labs[labs > 0]
            label_list = [int(l) for l in labs]
        except Exception:
            label_list = []
        # Initial draw should be RAW (no SG). Force sg_window=1 for grid render.
        grid_img = build_radial_profile_grid_image(
            df_all, peak_df=None, window_bins=int(smoothing), sg_window=1, sg_poly=int(sg_p), show_errorbars=bool(show_err), show_ratio=bool(show_ratio), labels=label_list, cols=3, tile_width=700, title_suffix=""
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
        # No SG table yet on initial compute; return RAW table and empty SG table
        return df_all, csv_all, None, None, grid_img, df_all, csv_all, grid_img, params
    
    run_prof_btn.click(
        fn=_radial_profile_cb,
        inputs=[tgt, ref, masks_state, tgt_chan, ref_chan, prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, profile_show_ratio, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps, components['sg_window'], components['sg_poly']],
        outputs=[profile_table_raw, profile_csv, profile_table_sg, profile_csv_sg, profile_plot, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
    )
    
    # Simple dict equality with fallback
    def params_equal(a, b):
        try:
            return a == b
        except Exception:
            return False

    def _radial_profile_single_or_all_cb(tgt_img, ref_img, masks, tchan, rchan, s, e, wsize, wstep, smoothing, show_err, show_ratio, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps, sg_w, sg_p, cache_df, cache_csv, cache_plot, cache_params, peak_df):
        bgm = str(bg_mode)
        mt = float(man_t) if (bg_en and bgm == "manual") else None
        mr = float(man_r) if (bg_en and bgm == "manual") else None
        # current masks/meta
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

        # If cached and params identical, rebuild grid and SG table from cache
        if (cache_df is not None) and params_equal(cache_params, cur_params):
            try:
                label_list_now = [int(l) for l in labs_now]
            except Exception:
                label_list_now = []
            title_suffix = "(SG)" if int(sg_w) >= 3 else "(RAW)"
            grid_img = build_radial_profile_grid_image(
                cache_df, peak_df, window_bins=int(smoothing), sg_window=int(sg_w), sg_poly=int(sg_p),
                show_errorbars=bool(show_err), show_ratio=bool(show_ratio), labels=label_list_now, cols=3, tile_width=700, title_suffix=title_suffix
            )
            df_sg = _apply_sg_to_profile_df(cache_df, int(sg_w), int(sg_p))
            tmp_csv_sg = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_sg.csv"); df_sg.to_csv(tmp_csv_sg.name, index=False)
            return cache_df, cache_csv, df_sg, tmp_csv_sg.name, grid_img, cache_df, cache_csv, grid_img, cur_params

        # Otherwise recompute RAW table, then build SG outputs and grid
        df_all, csv_all = radial_profile_all_cells(
            tgt_img, ref_img, masks, tchan, rchan,
            float(s), float(e), float(wsize), float(wstep),
            bool(bg_en), int(bg_r), bool(nm_en), nm_m,
            bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
            manual_tar_bg=mt, manual_ref_bg=mr, ratio_ref_epsilon=float(eps),
        )
        try:
            label_list_now = [int(l) for l in labs_now]
        except Exception:
            label_list_now = []
        title_suffix = "(SG)" if int(sg_w) >= 3 else "(RAW)"
        grid_img = build_radial_profile_grid_image(
            df_all, peak_df, window_bins=int(smoothing), sg_window=int(sg_w), sg_poly=int(sg_p),
            show_errorbars=bool(show_err), show_ratio=bool(show_ratio), labels=label_list_now, cols=3, tile_width=700, title_suffix=title_suffix
        )
        df_sg = _apply_sg_to_profile_df(df_all, int(sg_w), int(sg_p))
        tmp_csv_sg = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_sg.csv"); df_sg.to_csv(tmp_csv_sg.name, index=False)
        return df_all, csv_all, df_sg, tmp_csv_sg.name, grid_img, df_all, csv_all, grid_img, cur_params
    
    run_prof_single_btn.click(
        fn=_radial_profile_single_or_all_cb,
        inputs=[tgt, ref, masks_state, tgt_chan, ref_chan, prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, profile_show_ratio, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps, components['sg_window'], components['sg_poly'], prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state, peak_diff_state],
        outputs=[profile_table_raw, profile_csv, profile_table_sg, profile_csv_sg, profile_plot, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
    )

    # Helper: Apply SG smoothing per label to means (target/reference/ratio);
    # SEM/STD/Counts are preserved from RAW.
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
            out = pd.concat(parts, axis=0, ignore_index=True)
            return out.reset_index(drop=True)
        except Exception:
            return df.copy()
    
    # Peak difference
    def _peak_diff_cb(cached_df, quant_df, min_pct, max_pct, smoothing, show_err, show_ratio, sg_window, sg_poly, peak_algo, peak_slope_eps_rel):
        if cached_df is None or cached_df.empty:
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
        peak_df = compute_radial_peak_difference(cached_df, quant_df, float(min_pct), float(max_pct), algo=algo, sg_window=w, sg_poly=p, peak_slope_eps_rel=slope_rel)
        
        if peak_df.empty:
            return gr.update(value=pd.DataFrame()), None, gr.update(), None
        
        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_peak_difference.csv")
        peak_df.to_csv(tmp_csv.name, index=False)
        tmp_csv_path = tmp_csv.name
        tmp_csv.close()
        
        # Build grid reflecting peak markers
        try:
            labs = sorted(int(x) for x in cached_df["label"].dropna().unique())
        except Exception:
            labs = []
        try:
            grid_img = build_radial_profile_grid_image(cached_df, peak_df, window_bins=int(smoothing), sg_window=int(w), sg_poly=int(p), show_errorbars=bool(show_err), show_ratio=bool(show_ratio), labels=labs, cols=3, tile_width=700, title_suffix="(with peaks)")
        except Exception:
            grid_img = None
        
        return peak_df, tmp_csv_path, grid_img, peak_df
    
    run_peak_diff_btn.click(
        fn=_peak_diff_cb,
        inputs=[prof_cache_df_state, quant_df_state, peak_min_pct, peak_max_pct, prof_smoothing, prof_show_err, profile_show_ratio, components['sg_window'], components['sg_poly'], components['peak_algo'], components['peak_slope_eps_rel']],
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
        # Propagate to package-level so visualization uses the latest value
        try:
            dcq.LABEL_SCALE = float(v)
        except Exception:
            pass
        return None
    
    label_scale.change(fn=_set_label_scale, inputs=[label_scale], outputs=[])
    def _set_label_scale(v: float):
        # Propagate to package-level so visualization uses the latest value
        try:
            dcq.LABEL_SCALE = float(v)
        except Exception:
            pass
        return None
    
    label_scale.change(fn=_set_label_scale, inputs=[label_scale], outputs=[])
