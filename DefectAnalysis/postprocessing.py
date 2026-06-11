

import numpy as np
from typing import Dict, Tuple

# ── Class metadata ────────────────────────────────────────────────────────────
# Index 0 is background (not a defect).
CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "crazing",
    2: "inclusion",
    3: "patches",
    4: "pitted_surface",
    5: "rolled_in_scale",
    6: "scratches",
}

# Severity weight per defect class (higher = more dangerous).
# Tune these based on domain knowledge.
CLASS_WEIGHTS: Dict[int, float] = {
    0: 0.0,   # background — not a defect
    1: 0.9,   # crazing    — network of cracks, structurally serious
    2: 0.7,   # inclusion  — embedded foreign material
    3: 0.4,   # patches    — surface discolouration, less critical
    4: 0.6,   # pitted_surface
    5: 0.8,   # rolled_in_scale — lamination risk
    6: 0.5,   # scratches
}

# Severity thresholds (total defect area %).
SEVERITY_THRESHOLDS = {
    "Low":      (0.0,  5.0),
    "Medium":   (5.0,  15.0),
    "High":     (15.0, 30.0),
    "Critical": (30.0, 100.0),
}


# ── Core functions ────────────────────────────────────────────────────────────

def compute_defect_percentage(mask: np.ndarray) -> float:
    """
    Compute the percentage of pixels that contain any defect (class > 0).

    Parameters
    ----------
    mask : np.ndarray
        2-D integer array of shape (H, W). Values are class indices.

    Returns
    -------
    float
        Defect area as a percentage of the total image area (0–100).
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")

    total_pixels   = mask.size
    defect_pixels  = int(np.sum(mask > 0))
    return round(defect_pixels / total_pixels * 100.0, 4)


def compute_per_class_area(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute each defect class's area as a percentage of the total image.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer array of shape (H, W).

    Returns
    -------
    dict
        { class_name: area_percentage, ... }  — background excluded.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")

    total_pixels = mask.size
    result: Dict[str, float] = {}

    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_id == 0:          # skip background
            continue
        count = int(np.sum(mask == cls_id))
        result[cls_name] = round(count / total_pixels * 100.0, 4)

    return result


def classify_severity(defect_percent: float) -> str:
    """
    Classify overall defect severity based on total defect area percentage.

    Parameters
    ----------
    defect_percent : float
        Output of compute_defect_percentage().

    Returns
    -------
    str
        One of: "Low", "Medium", "High", "Critical"
    """
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= defect_percent < hi:
            return label
    return "Critical"   # fallback for 100 %


def compute_risk_score(mask: np.ndarray) -> float:
    """
    Compute a weighted risk score in [0, 100] that accounts for both
    defect area and defect type severity.

    Risk Score = Σ (class_area_% × class_weight)  capped at 100.

    Parameters
    ----------
    mask : np.ndarray
        2-D integer array of shape (H, W).

    Returns
    -------
    float
        Risk score between 0 and 100.
    """
    per_class = compute_per_class_area(mask)
    score = 0.0
    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        area   = per_class.get(cls_name, 0.0)
        weight = CLASS_WEIGHTS.get(cls_id, 0.5)
        score += area * weight

    return round(min(score, 100.0), 4)


def build_analysis_report(mask: np.ndarray) -> Dict:
    """
    Convenience wrapper — returns a complete analysis dictionary.

    Returns
    -------
    dict with keys:
        defect_percentage, per_class_area, severity, risk_score
    """
    defect_pct  = compute_defect_percentage(mask)
    per_class   = compute_per_class_area(mask)
    severity    = classify_severity(defect_pct)
    risk_score  = compute_risk_score(mask)

    return {
        "defect_percentage": defect_pct,
        "per_class_area":    per_class,
        "severity":          severity,
        "risk_score":        risk_score,
    }