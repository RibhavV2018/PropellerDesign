"""Phase 2 analysis: smoothness, feature importance, and cross-validation.

Three questions project-plan.md asks that the headline metrics do not answer.

SMOOTHNESS
    The plan warns that per-station prediction "can produce jagged,
    physically-invalid blade shapes". XGBParam exists to avoid that by
    construction. But the two currently score the same, so the warning is
    tested here rather than assumed. A blade is measured by the second
    difference of chord along the span -- curvature. Real blades set the
    reference: a prediction no rougher than a real propeller is smooth enough
    to manufacture, whatever its R^2.

FEATURE IMPORTANCE
    With three features this is quick, and the interesting question is how
    dominant CT is. If it carries nearly everything, that supports the claim
    that the problem is essentially one-dimensional once the physics is
    factored out -- which is also why linear regression competes.

    Measured by permutation rather than by XGBoost's gain. Two reasons: gain
    counts how often a feature was split on, which is not the same as how much
    the prediction depends on it, and xgboost 2.1.4 SEGFAULTS on
    feature_importances_ for models built with multi_strategy=
    "multi_output_tree". Permutation importance sidesteps both -- it shuffles
    one feature and measures how much the real-propeller error grows.

    READ THESE NUMBERS CAREFULLY. Permutation importance measures importance
    ON THE TEST DISTRIBUTION, not intrinsic importance. blade_count looks
    nearly irrelevant on real propellers (+2% chord MAE) purely because that
    set is 94% two-blade -- shuffling a near-constant column changes nothing.
    On balanced synthetic data the same model and the same feature give +129%,
    making blade count the single most important input for chord, exactly as
    solidity = B*c/(pi*R) implies.

    That gap is also the clearest justification for Phase 1b: the effect is
    large, and the measured data is structurally incapable of showing it.

CROSS-VALIDATION
    Five-fold on synthetic data, for model selection. Note this is NOT the
    train/test split: training is synthetic and testing is measured
    propellers, which is a stronger separation than any fold. Synthetic
    samples are independent draws, so plain KFold is right here -- the
    GroupKFold requirement applies to the real data, where 2121 rows are only
    119 blades.

    ./.venv/bin/python src/analyze_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import C_COLS, FEATURES, load_real, load_synthetic
from evaluate import ClassicalDesign, MeanBlade, geometric_metrics
from models import XGBParam, XGBRaw

N_CHORD = len(C_COLS)


def roughness(geom) -> np.ndarray:
    """RMS second difference of chord along the span -- curvature, per blade.

    A kink is a local spike in second difference. Summing over the blade gives
    one number per propeller that is comparable between predictions and real
    measured blades.
    """
    c = np.asarray(geom)[:, :N_CHORD]
    d2 = c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]
    return np.sqrt((d2 ** 2).mean(axis=1))


def main() -> None:
    Xs, ys, ps = load_synthetic()
    Xr, yr, _ = load_real()

    # --- smoothness ---------------------------------------------------------
    print("--- smoothness: RMS chord curvature per blade ---")
    print("    (real measured propellers are the reference for 'smooth enough')\n")

    real_rough = roughness(yr.values)
    rows = [("real propellers", real_rough)]
    fitted = {}
    for name, m in [("linear", LinearRegression()), ("xgb_raw", XGBRaw()),
                    ("xgb_param", XGBParam()), ("classical", ClassicalDesign()),
                    ("mean_blade", MeanBlade())]:
        if isinstance(m, XGBParam):
            m.fit(Xs.values, ys.values, params=ps.values)
        else:
            m.fit(Xs.values, ys.values)
        fitted[name] = m
        rows.append((name, roughness(m.predict(Xr.values))))

    print(f"{'':20}{'median':>10}{'90th pct':>11}{'max':>10}{'vs real':>10}")
    ref = np.median(real_rough)
    for name, r in rows:
        tag = "" if name == "real propellers" else f"{np.median(r)/ref:>9.2f}x"
        print(f"  {name:<18}{np.median(r):>10.5f}{np.percentile(r,90):>11.5f}"
              f"{r.max():>10.5f}{tag:>10}")

    # --- feature importance -------------------------------------------------
    print("\n--- permutation importance on REAL propellers ---")
    print("    (how much error grows when one feature is shuffled)\n")
    rng = np.random.default_rng(0)
    for mname in ["xgb_raw", "linear"]:
        m = fitted[mname]
        base = geometric_metrics(yr.values, m.predict(Xr.values))
        print(f"  {mname}:")
        for j, f in enumerate(FEATURES):
            Xp = Xr.values.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            g = geometric_metrics(yr.values, m.predict(Xp))
            dc = g["chord_MAE"] / base["chord_MAE"] - 1
            dt = g["twist_MAE_deg"] / base["twist_MAE_deg"] - 1
            print(f"    {f:<14} chord MAE {dc*100:>+7.1f}%   twist MAE {dt*100:>+7.1f}%")

    # --- cross-validation ---------------------------------------------------
    print("\n--- 5-fold CV on synthetic (model selection, not the real test) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    Xa, ya, pa = Xs.values, ys.values, ps.values

    print(f"{'':12}{'chord R2':>18}{'twist R2':>18}")
    for name, ctor in [("linear", lambda: LinearRegression()),
                       ("xgb_raw", lambda: XGBRaw()),
                       ("xgb_param", lambda: XGBParam())]:
        c_scores, t_scores = [], []
        for tr, te in kf.split(Xa):
            m = ctor()
            if isinstance(m, XGBParam):
                m.fit(Xa[tr], ya[tr], params=pa[tr])
            else:
                m.fit(Xa[tr], ya[tr])
            g = geometric_metrics(ya[te], m.predict(Xa[te]))
            c_scores.append(g["chord_R2"])
            t_scores.append(g["twist_R2"])
        print(f"  {name:<10}{np.mean(c_scores):>11.3f} +/- {np.std(c_scores):<5.3f}"
              f"{np.mean(t_scores):>11.3f} +/- {np.std(t_scores):<5.3f}")


if __name__ == "__main__":
    main()
