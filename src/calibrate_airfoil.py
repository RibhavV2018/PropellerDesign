"""Calibrate two airfoil constants on a held-out split of the propellers.

WHY A SPLIT, AND WHY THIS MATTERS HERE

The airfoil model has constants that physics alone does not pin down: how
much camber a generic propeller section has, and how much laminar separation
inflates its drag. Fitting them to measured data is legitimate -- calibrated
physics models are standard practice -- but only if the number you then
report comes from propellers the fit never saw.

Fitting on all 119 and then reporting the error on all 119 would be exactly
the original AIAA paper's mistake in a new costume: derive the labels from an
assumption, then report accuracy against those same labels. The split is what
keeps the reported figure honest.

Propellers are split, not rows. All ~17 operating points of one blade share a
geometry, so splitting rows would put the same propeller on both sides.

Only two constants are fitted, and they are nearly orthogonal:
    alpha0_deg  -- camber, drives the CT bias
    cd0_coeff   -- drag level, drives the CP bias
Everything else stays at its physics-derived value. Fewer knobs means less
opportunity to fit noise.

RESULT (SEED = 0, 71 calibration / 48 holdout propellers)

Calibration was run and NOT adopted. Best constants on the calibration set
were alpha0 = -5.5 deg, cd0_coeff = 8.0, against physics-derived defaults of
-3.0 and 6.2. On the holdout:

                                  med|eCT|  med|eCP|   CT sd   CP sd
    uncalibrated (physics)            13.5      14.0    18.2    23.8
    calibrated                        13.9      11.0    24.2    34.5

The typical case improves about 10% relative, entirely in CP, while the tails
get substantially worse -- CT does not improve at all. Extra camber pushes
more sections toward stall, which helps the average propeller and hurts the
small high-loading ones.

That is worth reporting as a finding rather than a failed step: the
physics-derived constants are already close to optimal, and the residual
error is dominated by not knowing each propeller's blade section, which no
amount of constant-tuning can recover. It also keeps the validation fully
independent, since nothing in the model was fitted to the measured data.

Rerun this if the airfoil model changes structurally -- the conclusion is
about these two constants, not about calibration in general.

    ./.venv/bin/python src/calibrate_airfoil.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bemt import make_airfoil
from validate_bemt import B_COLS, C_COLS, DATA, STATIONS
from bemt import solve_propeller

SEED = 0
CAL_FRACTION = 0.60
POINTS_PER_PROP = 3      # subsample for search speed; holdout uses everything


def split_propellers(df, seed=SEED, frac=CAL_FRACTION):
    """Split PROPELLERS (not rows) into calibration and holdout sets."""
    props = np.array(sorted(df.prop_name.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(props)
    cut = int(round(len(props) * frac))
    return set(props[:cut]), set(props[cut:])


def evaluate(df, airfoil):
    """Median absolute CT and CP error, in percent."""
    eCT, eCP = [], []
    for row in df.itertuples():
        try:
            out = solve_propeller(
                STATIONS,
                [getattr(row, c) for c in C_COLS],
                [getattr(row, b) for b in B_COLS],
                row.diameter_in, row.rpm, int(row.blade_count), airfoil=airfoil)
        except Exception:
            continue
        eCT.append((out["CT"] / row.CT - 1) * 100)
        eCP.append((out["CP"] / row.CP - 1) * 100)
    return np.array(eCT), np.array(eCP)


def objective(eCT, eCP):
    """Median ABSOLUTE error in both coefficients.

    Not |median error|. That penalizes bias only, so a search will happily
    trade a large increase in scatter for a small bias reduction -- which is
    exactly what happened on the first attempt here: bias halved while the
    holdout standard deviation went from 18% to 31%, producing a numerically
    "better" and physically worse model. Median absolute error prices both.
    """
    return float(np.median(np.abs(eCT)) + np.median(np.abs(eCP)))


def main() -> None:
    df = pd.read_csv(DATA)
    cal_props, hold_props = split_propellers(df)
    cal = df[df.prop_name.isin(cal_props)]
    hold = df[df.prop_name.isin(hold_props)]

    # Thin the calibration set for the search: a few operating points per
    # propeller is enough to locate the bias, and keeps the grid affordable.
    cal_small = cal.groupby("prop_name", group_keys=False).apply(
        lambda g: g.iloc[:: max(1, len(g) // POINTS_PER_PROP)][:POINTS_PER_PROP])

    print(f"calibration: {len(cal_props)} propellers ({len(cal)} points, "
          f"{len(cal_small)} used for search)")
    print(f"holdout    : {len(hold_props)} propellers ({len(hold)} points)  "
          f"-- never seen during fitting\n")

    best = None
    for a0 in np.arange(-10.0, 2.01, 0.5):
        for cd0 in np.arange(3.0, 12.01, 0.5):
            eCT, eCP = evaluate(cal_small, make_airfoil(alpha0_deg=a0, cd0_coeff=cd0))
            score = objective(eCT, eCP)
            if best is None or score < best[0]:
                best = (score, a0, cd0)
    _, a0, cd0 = best
    print(f"best on calibration set: alpha0 = {a0:+.1f} deg, cd0_coeff = {cd0:.1f}\n")

    fitted = make_airfoil(alpha0_deg=a0, cd0_coeff=cd0)
    print(f"{'':28}{'CT med':>9}{'CT sd':>8}{'CP med':>9}{'CP sd':>8}")
    for label, subset, af in [
        ("uncalibrated, holdout", hold, make_airfoil()),
        ("calibrated,   calibration", cal, fitted),
        ("calibrated,   HOLDOUT", hold, fitted),
    ]:
        eCT, eCP = evaluate(subset, af)
        print(f"  {label:<26}{np.median(eCT):>+9.1f}{eCT.std():>8.1f}"
              f"{np.median(eCP):>+9.1f}{eCP.std():>8.1f}")

    print(f"\nThe HOLDOUT row is the number to report: {len(hold_props)} propellers "
          f"the calibration never saw.")
    print(f"Reproduce with SEED = {SEED}, CAL_FRACTION = {CAL_FRACTION}.")


if __name__ == "__main__":
    main()
