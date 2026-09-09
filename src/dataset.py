"""Features and targets for the Phase 2 inverse-design model.

THE TASK

    user gives:   target thrust, diameter, rpm, blade count
    model gives:  chord and twist at 18 radial stations

Both sides are made DIMENSIONLESS. The user's three dimensional inputs
collapse into a thrust coefficient:

    CT = T / (rho * n^2 * D^4)

and the geometry targets are already normalized (c/R, and twist in degrees).
That matters with only 119 independent real blades: asking a tree ensemble to
rediscover the n^2 D^4 relationship from data spends scarce samples on physics
that physics.py already knows exactly. Handing it CT spends them on the part
that is genuinely unknown -- the shape.

THE REYNOLDS TRAP

The obvious third feature is section Reynolds number, which is what drives the
low-Re behaviour the whole solver was built around. But the honest definition,

    Re_75 = rho * V_75 * chord_75 / mu

needs the CHORD, which is what the model is trying to predict. Using it would
leak the answer into the question, and would be uncomputable at inference time
when the user has no geometry yet.

So the feature is a chord-free rotational Reynolds number, built from tip
radius instead of chord:

    Re_ref = rho * V_75 * R / mu

It is monotone in the true section Reynolds number (chord ratios cluster
around 0.15), computable from diameter and rpm alone, and carries the same
scale information without the circularity.

EVERY FEATURE IS COMPUTABLE FROM THE USER'S OWN INPUTS. That is the test a
feature has to pass here, and it is easy to fail by accident.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bemt import IN_M, MU, RHO

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
REAL = PROC / "uiuc_hover.csv"
SYNTH = PROC / "synthetic_frontier.csv"

STATIONS = [round(0.15 + 0.05 * k, 2) for k in range(18)]
C_COLS = [f"c_R_r{int(round(x * 100)):03d}" for x in STATIONS]
B_COLS = [f"beta_r{int(round(x * 100)):03d}" for x in STATIONS]

FEATURES = ["CT", "blade_count", "log_Re_ref"]
TARGETS = C_COLS + B_COLS          # 36 outputs

# The generator's own parameters. Available for synthetic data only -- real
# propellers were never built from them. Predicting these five and
# reconstructing the blade is the smooth alternative to predicting 36 raw
# station values, and guarantees a physically valid shape by construction.
PARAMS = ["pitch_ratio", "solidity", "washout_deg", "shape_c1", "shape_c2"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the model features. Requires diameter_in, rpm, CT, blade_count."""
    out = df.copy()
    R = out.diameter_in * IN_M / 2.0
    v75 = 2.0 * np.pi * (out.rpm / 60.0) * 0.75 * R
    out["Re_ref"] = RHO * v75 * R / MU
    out["log_Re_ref"] = np.log10(out.Re_ref)
    return out


def load_synthetic(path=SYNTH):
    """Training data: the efficient frontier of BEMT-generated propellers.

    Returns:
        (X, y, params) -- features, the 36 geometry targets, and the five
        generator parameters for the smooth-representation comparison.
    """
    df = add_features(pd.read_csv(path))
    return df[FEATURES], df[TARGETS], df[PARAMS]


def load_real(path=REAL):
    """Test data: measured propellers.

    Returns:
        (X, y, groups) -- groups is prop_name, and every split MUST use it.
        The 2121 rows are only 119 independent blades; each propeller's ~17
        operating points carry an identical copy of the geometry being
        predicted, so a random row split trains and tests on the same blade.
    """
    df = add_features(pd.read_csv(path))
    return df[FEATURES], df[TARGETS], df["prop_name"]


def main() -> None:
    Xs, ys, ps = load_synthetic()
    Xr, yr, gr = load_real()

    print(f"synthetic (train): {Xs.shape[0]:,} blades   "
          f"{Xs.shape[1]} features -> {ys.shape[1]} targets")
    print(f"real (test)      : {Xr.shape[0]:,} rows across {gr.nunique()} propellers")

    print("\nfeature ranges:")
    print(f"{'':14}{'synthetic':>26}{'real':>26}")
    for f in FEATURES:
        print(f"  {f:<12}{Xs[f].min():>11.3f} - {Xs[f].max():<12.3f}"
              f"{Xr[f].min():>11.3f} - {Xr[f].max():<12.3f}")

    # Train/test overlap: the model must interpolate on real data, not
    # extrapolate. Anything outside the synthetic range is untrainable.
    print("\nreal rows outside the synthetic feature range:")
    for f in FEATURES:
        out = ((Xr[f] < Xs[f].min()) | (Xr[f] > Xs[f].max())).mean() * 100
        print(f"  {f:<14}{out:5.1f}%")


if __name__ == "__main__":
    main()
