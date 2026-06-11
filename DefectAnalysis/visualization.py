

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

# ── Colour palette (BGR) — one colour per class ───────────────────────────────
# Index 0 = background (transparent / not drawn).
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0,   0,   0),    # background  — unused
    1: (0,   0,   255),  # crazing     — red
    2: (0,   255, 255),  # inclusion   — yellow
    3: (0,   165, 255),  # patches     — orange
    4: (255, 0,   0),    # pitted_surface — blue
    5: (0,   255, 0),    # rolled_in_scale — green
    6: (255, 0,   255),  # scratches   — magenta
}

ALPHA: float = 0.45   # blend strength for mask overlay


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = ALPHA,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Overlay a coloured segmentation mask on the original image.

    Parameters
    ----------
    image : np.ndarray
        BGR image of shape (H, W, 3) or grayscale (H, W).
    mask : np.ndarray
        2-D integer array of shape (H, W), values = class indices.
    alpha : float
        Opacity of the mask layer (0 = invisible, 1 = opaque).
    colors : dict, optional
        Override default CLASS_COLORS_BGR.

    Returns
    -------
    np.ndarray
        BGR image of shape (H, W, 3) with coloured defects overlaid.
    """
    palette = colors if colors else CLASS_COLORS_BGR

    # Ensure 3-channel BGR
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy().astype(np.uint8)

    # Resize mask to match image if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    colour_layer = np.zeros_like(overlay)
    has_defect   = False

    for cls_id, color in palette.items():
        if cls_id == 0:
            continue
        region = mask == cls_id
        if not region.any():
            continue
        has_defect = True
        colour_layer[region] = color

    if has_defect:
        defect_pixels       = mask > 0
        overlay[defect_pixels] = cv2.addWeighted(
            overlay, 1 - alpha,
            colour_layer, alpha,
            0,
        )[defect_pixels]

    return overlay


def generate_heatmap(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Generate a heatmap from the defect mask (any non-background pixel = heat).

    Parameters
    ----------
    mask : np.ndarray
        2-D integer array (H, W).
    image : np.ndarray, optional
        If provided, blend the heatmap with the original image.
    colormap : int
        OpenCV colormap constant (default: COLORMAP_JET).

    Returns
    -------
    np.ndarray
        BGR heatmap image (H, W, 3).
    """
    binary = (mask > 0).astype(np.uint8) * 255

    # Smooth for a softer heatmap look
    blurred   = cv2.GaussianBlur(binary, (15, 15), 0)
    heatmap   = cv2.applyColorMap(blurred, colormap)

    if image is not None:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[:2] != heatmap.shape[:2]:
            image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
        heatmap = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return heatmap


def draw_legend(
    image: np.ndarray,
    class_names: Dict[int, str],
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Draw a colour legend on the top-left corner of the image.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    class_names : dict
        { class_id: class_name }
    colors : dict, optional
        Override default CLASS_COLORS_BGR.

    Returns
    -------
    np.ndarray
        Image with legend drawn.
    """
    palette = colors if colors else CLASS_COLORS_BGR
    out     = image.copy()
    x, y    = 10, 20
    step    = 22

    for cls_id, name in class_names.items():
        if cls_id == 0:
            continue
        color = palette.get(cls_id, (255, 255, 255))
        cv2.rectangle(out, (x, y - 12), (x + 16, y + 4), color, -1)
        cv2.putText(
            out, name, (x + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        y += step

    return out


def draw_analysis_text(
    image: np.ndarray,
    defect_pct: float,
    severity: str,
    risk_score: float,
    rul_days: int,
) -> np.ndarray:
    """
    Burn analysis summary text onto the bottom of the image.
    """
    out = image.copy()
    h   = out.shape[0]

    severity_color = {
        "Low":      (0,   200, 0),
        "Medium":   (0,   165, 255),
        "High":     (0,   60,  255),
        "Critical": (0,   0,   255),
    }.get(severity, (255, 255, 255))

    lines = [
        (f"Defect Area : {defect_pct:.2f}%",  (200, 200, 200)),
        (f"Severity    : {severity}",          severity_color),
        (f"Risk Score  : {risk_score:.1f}/100",(200, 200, 200)),
        (f"Est. RUL    : {rul_days} days",     (200, 200, 200)),
    ]

    for i, (text, color) in enumerate(reversed(lines)):
        y = h - 12 - i * 22
        cv2.putText(
            out, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    return out