
import numpy as np
from typing import Dict, Tuple, Optional
from .postprocessing import CLASS_NAMES


# ── Rule-based constants ──────────────────────────────────────────────────────

# Baseline RUL (days) for a pristine surface with zero defects.
BASELINE_RUL_DAYS: float = 365.0

# Maximum RUL reduction triggered by risk score alone (days).
MAX_RISK_PENALTY: float = 300.0

# Extra penalty (days) per unit of critical-class area percentage.
CRITICAL_CLASS_EXTRA_PENALTY: float = 5.0

# Classes considered structurally critical (by class id).
CRITICAL_CLASS_IDS = {1, 5}   # crazing, rolled_in_scale


def estimate_remaining_life(
    defect_percent: float,
    risk_score: float,
    per_class_area: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Rule-based RUL estimation.

    Logic
    -----
    1. Start from BASELINE_RUL_DAYS.
    2. Subtract a penalty proportional to risk_score.
    3. Apply an additional penalty for each critical defect class area.
    4. Apply a floor of 0 days.

    Parameters
    ----------
    defect_percent : float
        Total defect area percentage (0–100).
    risk_score : float
        Weighted risk score (0–100) from compute_risk_score().
    per_class_area : dict, optional
        Output of compute_per_class_area(). Used for critical-class penalty.

    Returns
    -------
    dict
        {
            "rul_days":          int,
            "rul_recommendation": str,
            "method":            "rule_based",
        }
    """
    # Base penalty from risk score (linear)
    risk_penalty = (risk_score / 100.0) * MAX_RISK_PENALTY

    # Extra penalty for structurally critical classes
    critical_penalty = 0.0
    if per_class_area:
        for cls_id in CRITICAL_CLASS_IDS:
            cls_name = CLASS_NAMES.get(cls_id, "")
            area     = per_class_area.get(cls_name, 0.0)
            critical_penalty += area * CRITICAL_CLASS_EXTRA_PENALTY

    rul = BASELINE_RUL_DAYS - risk_penalty - critical_penalty
    rul = max(0.0, rul)

    recommendation = _rul_recommendation(rul, risk_score)

    return {
        "rul_days":           int(rul),
        "rul_recommendation": recommendation,
        "method":             "rule_based",
    }


def _rul_recommendation(rul_days: float, risk_score: float) -> str:
    """Human-readable maintenance recommendation."""
    if rul_days == 0 or risk_score >= 80:
        return ("🔴 IMMEDIATE ACTION REQUIRED: Surface has exceeded safe limits. "
                "Replace or withdraw from service immediately.")
    elif rul_days < 30 or risk_score >= 60:
        return ("🟠 URGENT: Schedule maintenance within 30 days. "
                "Increase inspection frequency.")
    elif rul_days < 90 or risk_score >= 40:
        return ("🟡 MONITOR CLOSELY: Plan maintenance within the next quarter. "
                "Continue regular inspections.")
    else:
        return ("🟢 ACCEPTABLE: Surface is within operational limits. "
                "Maintain standard inspection schedule.")


# ── ML-based RUL (scikit-learn) ───────────────────────────────────────────────

def build_ml_rul_model():
    """
    Build and return a trained scikit-learn RUL regression model.

    In production, replace the synthetic training data below with real
    historical records: (defect features) → measured RUL.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted model ready for predict().
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        raise ImportError(
            "scikit-learn is required for ML-based RUL. "
            "Install with: pip install scikit-learn"
        )

    # ── Synthetic training data ───────────────────────────────────────────────
    # Features: [defect_pct, risk_score, crazing_%, inclusion_%, patches_%,
    #            pitted_%, rolled_in_scale_%, scratches_%]
    rng = np.random.default_rng(seed=42)
    n   = 500

    defect_pct   = rng.uniform(0, 60, n)
    risk_score   = rng.uniform(0, 100, n)
    class_areas  = rng.dirichlet(np.ones(6), n) * defect_pct[:, None]

    X = np.column_stack([defect_pct, risk_score, class_areas])

    # Simple physics-informed target (replace with real data)
    y = np.clip(
        BASELINE_RUL_DAYS
        - (risk_score / 100.0) * MAX_RISK_PENALTY
        - class_areas[:, 0] * CRITICAL_CLASS_EXTRA_PENALTY   # crazing
        - class_areas[:, 4] * CRITICAL_CLASS_EXTRA_PENALTY   # rolled_in_scale
        + rng.normal(0, 10, n),                               # noise
        0, BASELINE_RUL_DAYS,
    )
    # ─────────────────────────────────────────────────────────────────────────

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr",    GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42,
        )),
    ])
    model.fit(X, y)
    return model


def estimate_rul_ml(
    defect_percent: float,
    risk_score: float,
    per_class_area: Dict[str, float],
    ml_model=None,
) -> Dict:
    """
    ML-based RUL estimation using a trained regression model.

    Parameters
    ----------
    defect_percent : float
    risk_score : float
    per_class_area : dict
    ml_model : fitted sklearn Pipeline, optional
        If None, a new model is built on synthetic data.

    Returns
    -------
    dict
        { "rul_days": int, "rul_recommendation": str, "method": "ml_based" }
    """
    if ml_model is None:
        ml_model = build_ml_rul_model()

    class_order = ["crazing", "inclusion", "patches",
                   "pitted_surface", "rolled_in_scale", "scratches"]
    class_feat  = [per_class_area.get(c, 0.0) for c in class_order]

    features = np.array([[defect_percent, risk_score, *class_feat]])
    rul      = float(np.clip(ml_model.predict(features)[0], 0, BASELINE_RUL_DAYS))

    return {
        "rul_days":           int(rul),
        "rul_recommendation": _rul_recommendation(rul, risk_score),
        "method":             "ml_based",
    }