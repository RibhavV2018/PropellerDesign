"""The tool itself: requirements in, propeller geometry out.

Implements SCOPE.md's input/output contract.

    IN   target thrust, max diameter, target rpm, blade count (OPTIONAL)
    OUT  chord and twist at 18 stations, blade count used, predicted thrust
         and mechanical shaft power, and thrust per shaft watt

Three stages, each already built and validated separately:

    physics.py   thrust, diameter, rpm  ->  CT          (dimensionless)
    the model    CT, blade count, Re    ->  geometry    (Phase 2)
    bemt.py      geometry               ->  what it actually does (Phase 1b)

The third stage is what makes this a design tool rather than a regression.
The model proposes a blade; the solver reports what that blade really does, so
the numbers handed back are simulated performance, not the model's own
optimistic estimate of itself.

BLADE COUNT AUTO-SELECTION

When blade_count is None, every candidate from 2 to 6 is designed and
simulated, and the one with the best thrust per shaft watt that still meets
the thrust requirement wins. SCOPE.md calls this a thin wrapper over model
inference, and it is -- but note the thrust constraint is doing real work.
Efficiency alone is maximized by a blade that produces almost no thrust, since
T/P grows without bound as thrust goes to zero. Ranking on efficiency without
a thrust floor selects exactly those.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bemt import IN_M, MU, RHO, low_re_airfoil, solve_propeller
from dataset import C_COLS, STATIONS
from physics import GF_PER_N

N_CHORD = len(C_COLS)
CANDIDATE_BLADES = (2, 3, 4, 5, 6)
THRUST_TOLERANCE = 0.15      # a candidate must land within 15% of the target


def features_for(thrust_N, diameter_in, rpm):
    """Build the model's three dimensionless inputs from user requirements.

    Every one is computable from what the user supplies -- that is the test a
    feature here has to pass, and Re_ref exists because the honest section
    Reynolds number would need the chord this is trying to predict.
    """
    n = rpm / 60.0
    D = diameter_in * IN_M
    ct = thrust_N / (RHO * n**2 * D**4)

    R = D / 2.0
    v75 = 2.0 * np.pi * n * 0.75 * R
    re_ref = RHO * v75 * R / MU
    return ct, np.log10(re_ref)


def design(model, thrust_N, diameter_in, rpm, blade_count=None):
    """Design a propeller for one requirement.

    Args:
        model: a fitted Phase 2 model with .predict(X).
        thrust_N: target thrust at the design point, newtons.
        diameter_in: propeller diameter, inches.
        rpm: target shaft speed.
        blade_count: fixed count, or None to auto-select from 2-6.

    Returns:
        dict with c_R, beta_deg, blade_count, thrust_N, shaft_power_W,
        thrust_per_watt_gf_W, CT, CP, meets_thrust -- plus `candidates` when
        the blade count was auto-selected.

        Performance figures are SIMULATED from the predicted geometry, not
        predicted directly.
    """
    ct_target, log_re = features_for(thrust_N, diameter_in, rpm)
    counts = [blade_count] if blade_count else CANDIDATE_BLADES

    results = []
    for B in counts:
        pred = model.predict(np.array([[ct_target, float(B), log_re]]))[0]
        c_R, beta = list(pred[:N_CHORD]), list(pred[N_CHORD:])
        try:
            out = solve_propeller(STATIONS, c_R, beta, diameter_in, rpm, B,
                                  airfoil=low_re_airfoil)
        except Exception:
            continue
        if out["shaft_power_W"] <= 0:
            continue

        results.append({
            "c_R": c_R, "beta_deg": beta, "blade_count": B,
            "thrust_N": out["thrust_N"], "shaft_power_W": out["shaft_power_W"],
            "thrust_per_watt_gf_W": out["thrust_N"] * GF_PER_N / out["shaft_power_W"],
            "CT": out["CT"], "CP": out["CP"],
            "thrust_error_pct": (out["thrust_N"] / thrust_N - 1) * 100,
            "meets_thrust": abs(out["thrust_N"] / thrust_N - 1) <= THRUST_TOLERANCE,
        })

    if not results:
        raise RuntimeError("no candidate blade could be solved")

    # Prefer candidates that actually meet the requirement; only if none do,
    # fall back to whichever comes closest, so the tool always answers and the
    # caller can see it fell short via meets_thrust.
    viable = [r for r in results if r["meets_thrust"]]
    if viable:
        best = max(viable, key=lambda r: r["thrust_per_watt_gf_W"])
    else:
        best = min(results, key=lambda r: abs(r["thrust_error_pct"]))

    if blade_count is None:
        best = dict(best, candidates=results)
    return best
