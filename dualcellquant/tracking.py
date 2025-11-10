"""
Tracking and ID preservation utilities.

Goal: Maintain stable cell IDs across frames by matching current Cellpose
labels to a previous frame's labels using IoU-based Hungarian assignment.

APIs:
- relabel_to_previous(prev_labels, curr_labels, iou_threshold=0.2) -> (relabels, mapping_df, tiff_path)
- track_sequence(label_list, iou_threshold=0.2) -> list of relabeled arrays, list of mapping DataFrames

Notes:
- Handles births/deaths automatically. Splits/merges are resolved by keeping
  the best IoU match and assigning new IDs to remaining unmatched components.
- Does not mutate input arrays.
"""

from __future__ import annotations

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .visualization import save_label_tiff


def _compute_overlap_iou(prev: np.ndarray, curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute intersection counts matrix and IoU matrix between labeled images.

    Args:
        prev: 2D label image (int), previous frame labels, 0 = background.
        curr: 2D label image (int), current frame labels, 0 = background.

    Returns:
        inter: (P+1, C+1) intersection counts for label pairs (including 0)
        iou: (P, C) IoU per (prev_label>=1, curr_label>=1)
        areas: tuple (area_prev[1..P], area_curr[1..C])
    """
    if prev.shape != curr.shape:
        raise ValueError(f"Shape mismatch: prev {prev.shape} vs curr {curr.shape}")
    prev = prev.astype(np.int64, copy=False)
    curr = curr.astype(np.int64, copy=False)
    pmax = int(prev.max())
    cmax = int(curr.max())
    if pmax == 0 or cmax == 0:
        inter = np.zeros((pmax + 1, cmax + 1), dtype=np.int64)
        iou = np.zeros((pmax, cmax), dtype=np.float64)
        return inter, iou, (np.zeros(pmax + 1, dtype=np.int64), np.zeros(cmax + 1, dtype=np.int64))

    # Use unique pair counting to build intersection matrix efficiently
    base = cmax + 1
    flat_prev = prev.ravel()
    flat_curr = curr.ravel()
    key = flat_prev.astype(np.int64) * base + flat_curr.astype(np.int64)
    counts = np.bincount(key)
    # Build full matrix (P+1, C+1)
    inter = np.zeros((pmax + 1, cmax + 1), dtype=np.int64)
    nonzero_idx = np.flatnonzero(counts)
    pv = (nonzero_idx // base).astype(np.int64)
    cv = (nonzero_idx % base).astype(np.int64)
    inter[pv, cv] = counts[nonzero_idx]

    area_prev = inter.sum(axis=1)  # length P+1
    area_curr = inter.sum(axis=0)  # length C+1

    # Exclude background row/col for IoU
    inter_pc = inter[1:, 1:].astype(np.float64)
    area_prev_v = area_prev[1:].astype(np.float64)
    area_curr_v = area_curr[1:].astype(np.float64)
    # Broadcast union = ap + ac - inter
    union = (area_prev_v[:, None] + area_curr_v[None, :] - inter_pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter_pc / union, 0.0)
    return inter, iou, (area_prev, area_curr)


def relabel_to_previous(
    prev_labels: np.ndarray,
    curr_labels: np.ndarray,
    *,
    iou_threshold: float = 0.2,
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """
    Relabel current labels to preserve previous IDs using IoU+Hungarian assignment.

    Unmatched current objects receive new IDs starting from max(prev)+1.

    Returns:
        relabeled: np.ndarray with same shape as curr_labels, labels remapped
        mapping_df: DataFrame with columns [curr_id, assigned_prev_id, IoU]
        tiff_path: path to a temporary TIFF file of relabeled labels
    """
    prev = np.asarray(prev_labels)
    curr = np.asarray(curr_labels)
    if prev.shape != curr.shape:
        raise ValueError("prev_labels and curr_labels must have the same shape")
    pmax = int(prev.max())
    cmax = int(curr.max())
    if cmax == 0:
        # Nothing to relabel
        rel = np.zeros_like(curr, dtype=curr.dtype)
        df = pd.DataFrame(columns=["curr_id", "assigned_prev_id", "IoU"])  # empty
        tiff = save_label_tiff(rel, "relabel")
        return rel, df, tiff

    _, iou, _ = _compute_overlap_iou(prev, curr)
    # Cost = 1 - IoU; we will mask low IoU by setting large cost
    P, C = iou.shape
    cost = 1.0 - iou
    # Apply threshold gating
    gate = iou >= float(iou_threshold)
    # For pairs below threshold, set cost to a large number so they won't be chosen
    large = 10.0
    cost_gated = np.where(gate, cost, large)

    # Hungarian works on square; pad if necessary
    n = max(P, C)
    pad = np.full((n, n), large, dtype=np.float64)
    pad[:P, :C] = cost_gated
    row_ind, col_ind = linear_sum_assignment(pad)

    # Build mapping curr_id -> prev_id for valid matches
    mapping: Dict[int, int] = {}
    iou_map: Dict[int, float] = {}
    for r, c in zip(row_ind, col_ind):
        if r < P and c < C:
            iou_rc = iou[r, c]
            if iou_rc >= float(iou_threshold):
                prev_id = r + 1  # because rows 0..P-1 correspond to prev labels 1..P
                curr_id = c + 1  # cols 0..C-1 correspond to curr labels 1..C
                mapping[curr_id] = prev_id
                iou_map[curr_id] = float(iou_rc)

    # Assign new IDs to unmatched current objects
    relabeled = np.zeros_like(curr, dtype=np.int32)
    next_id = pmax + 1
    rows = []
    for curr_id in range(1, cmax + 1):
        mask = (curr == curr_id)
        if not mask.any():
            continue
        if curr_id in mapping:
            pid = mapping[curr_id]
            relabeled[mask] = pid
            rows.append({"curr_id": int(curr_id), "assigned_prev_id": int(pid), "IoU": float(iou_map.get(curr_id, 0.0))})
        else:
            relabeled[mask] = next_id
            rows.append({"curr_id": int(curr_id), "assigned_prev_id": int(next_id), "IoU": 0.0})
            next_id += 1

    df = pd.DataFrame(rows).sort_values("curr_id").reset_index(drop=True)
    tiff = save_label_tiff(relabeled, "relabel")
    return relabeled, df, tiff


def track_sequence(
    labels_list: List[np.ndarray],
    *,
    iou_threshold: float = 0.2,
) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
    """
    Apply relabel_to_previous iteratively over a sequence of label images.

    The first frame is returned as-is. Each subsequent frame is relabeled
    to preserve IDs from the immediately preceding relabeled frame.
    """
    if labels_list is None or len(labels_list) == 0:
        return [], []
    out_labels: List[np.ndarray] = []
    mappings: List[pd.DataFrame] = []
    prev = labels_list[0].astype(np.int32, copy=False)
    out_labels.append(prev)
    mappings.append(pd.DataFrame(columns=["curr_id", "assigned_prev_id", "IoU"]))
    for i in range(1, len(labels_list)):
        cur = labels_list[i].astype(np.int32, copy=False)
        rel, df, _ = relabel_to_previous(prev, cur, iou_threshold=iou_threshold)
        out_labels.append(rel)
        mappings.append(df)
        prev = rel
    return out_labels, mappings
