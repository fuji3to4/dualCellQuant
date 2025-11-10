"""
DualCellQuant: Dual-channel fluorescence cell quantification tool.

Main modules:
- core: Segmentation, mask application, and quantification
- radial: Radial profile analysis and peak detection
- visualization: Overlay generation and plotting utilities
"""

__version__ = "0.2.0"

# Import constants
LABEL_SCALE = 1.8

# Core functions
from .core import (
    get_model,
    pil_to_numpy,
    pil_to_numpy_native,
    background_correction,
    normalize_array,
    preprocess_for_processing,
    compute_dark_background,
    extract_single_channel,
    global_threshold_mask,
    per_cell_threshold,
    cleanup_mask,
    run_segmentation,
    apply_mask,
    integrate_and_quantify,
)

# Radial analysis functions
from .radial import (
    radial_mask,
    radial_profile_analysis,
    radial_profile_single,
    radial_profile_all_cells,
    compute_radial_peak_difference,
)

# Visualization functions
from .visualization import (
    colorize_overlay,
    vivid_label_image,
    annotate_ids,
    arr01_to_pil_for_preview,
    save_bool_mask_tiff,
    save_label_tiff,
    plot_radial_profile_with_peaks,
    save_radial_profile_grid_png,
    build_radial_profile_grid_image,
)

# Tracking functions
from .tracking import (
    relabel_to_previous,
    track_sequence,
)

# UI function
from .ui import build_ui

__all__ = [
    # Constants
    "LABEL_SCALE",
    # Core
    "get_model",
    "pil_to_numpy",
    "pil_to_numpy_native",
    "background_correction",
    "normalize_array",
    "preprocess_for_processing",
    "compute_dark_background",
    "extract_single_channel",
    "global_threshold_mask",
    "per_cell_threshold",
    "cleanup_mask",
    "run_segmentation",
    "apply_mask",
    "integrate_and_quantify",
    # Radial
    "radial_mask",
    "radial_profile_analysis",
    "radial_profile_single",
    "radial_profile_all_cells",
    "compute_radial_peak_difference",
    # Visualization
    "colorize_overlay",
    "vivid_label_image",
    "annotate_ids",
    "arr01_to_pil_for_preview",
    "save_bool_mask_tiff",
    "save_label_tiff",
    "plot_radial_profile_with_peaks",
    "save_radial_profile_grid_png",
    "build_radial_profile_grid_image",
    # Tracking
    "relabel_to_previous",
    "track_sequence",
    # UI
    "build_ui",
]
