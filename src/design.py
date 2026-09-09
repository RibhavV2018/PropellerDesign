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
simulated, and the most efficient one that meets the thrust requirement wins.
SCOPE.md calls this a thin wrapper over model inference, and it is -- but the
thrust constraint is doing real work. Efficiency alone is maximized by a blade
producing almost no thrust, since T/P grows without bound as thrust goes to
zero. Ranking on efficiency without a thrust floor selects exactly those.

WHY THE ACCEPTABLE BAND IS ASYMMETRIC

The band was once +/-15%, which sounds even-handed and is not: measured over a
sweep of realistic requirements it undershot the target 66% of the time, by up
to 15%.

Undershoot and overshoot are not symmetric failures here. SCOPE.md derives the
target rpm from motor Kv times battery voltage -- a no-load CEILING, not a set
point, and the loaded motor turns slower still. So rpm is not a knob the user
can turn up to recover missing thrust. A design that falls short at that rpm
cannot reach the target at all, which is a hard failure rather than a
performance shortfall. Overshoot merely costs weight and current.

So the default requires thrust at or above target, and caps overshoot at +20%
because an over-thrusting propeller is heavier, higher-inertia, and draws more
current than the mission needs. Measured cost of the change: about 3.8% median
efficiency, and no qualifying candidate in roughly 22% of cases -- which falls
back to the closest and says so.

TWO HONEST CAVEATS

The solver UNDER-predicts thrust: Phase 1b measured CT error at -12.6% median
against wind tunnel data, so a real blade built to a design that simulates at
exactly target would likely produce more. This default is therefore doubly
conservative. That is a reason to expose the setting, not to rely on it, since
the scatter around that bias is 17-19%.

And the prediction carries roughly 5% model error on top of that. A 0%
boundary is more precise than the number it constrains -- it is a bias in the
right direction, not a guarantee.
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

# Acceptable thrust band, as ratios of the target. Asymmetric on purpose --
# see the module docstring. MIN_THRUST_RATIO = 1.0 means "must meet the
# requirement"; lower it if the target rpm genuinely has headroom.
MIN_THRUST_RATIO = 1.00
MAX_THRUST_RATIO = 1.20


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


def design(model, thrust_N, diameter_in, rpm, blade_count=None,
           min_thrust_ratio=MIN_THRUST_RATIO, max_thrust_ratio=MAX_THRUST_RATIO):
    """Design a propeller for one requirement.

    Args:
        model: a fitted Phase 2 model with .predict(X).
        thrust_N: target thrust at the design point, newtons.
        diameter_in: propeller diameter, inches.
        rpm: target shaft speed.
        blade_count: fixed count, or None to auto-select from 2-6.
        min_thrust_ratio: lowest acceptable thrust as a fraction of target.
            1.0 refuses to undershoot. Lower it when the target rpm has real
            headroom, or to trade thrust margin for efficiency.
        max_thrust_ratio: highest acceptable thrust as a fraction of target.

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
        # Cast to native Python types here rather than at every call site.
        # numpy scalars leak out otherwise: np.bool_ reports its type name as
        # "bool" but json.dumps rejects it, and np.float64 the same -- which
        # surfaces as an opaque 500 in any caller that serializes the result.
        c_R = [float(v) for v in pred[:N_CHORD]]
        beta = [float(v) for v in pred[N_CHORD:]]
        try:
            out = solve_propeller(STATIONS, c_R, beta, diameter_in, rpm, B,
                                  airfoil=low_re_airfoil)
        except Exception:
            continue
        if out["shaft_power_W"] <= 0:
            continue

        err = float(out["thrust_N"] / thrust_N - 1)
        results.append({
            "c_R": c_R, "beta_deg": beta, "blade_count": int(B),
            "thrust_N": float(out["thrust_N"]),
            "shaft_power_W": float(out["shaft_power_W"]),
            "thrust_per_watt_gf_W": float(out["thrust_N"] * GF_PER_N / out["shaft_power_W"]),
            "CT": float(out["CT"]), "CP": float(out["CP"]),
            "thrust_error_pct": err * 100,
            "meets_thrust": bool(min_thrust_ratio <= 1.0 + err <= max_thrust_ratio),
        })

    if not results:
        raise RuntimeError("no candidate blade could be solved")

    # Three tiers, in order.
    #
    #   1. Inside the acceptable band -> the most efficient one. This is
    #      SCOPE.md's rule: maximize thrust per watt subject to the constraint.
    #   2. Nothing in the band, but something exceeds the target -> the
    #      SMALLEST such candidate. Overshooting past the cap costs weight and
    #      current; undershooting can mean the aircraft never leaves the
    #      ground, because the target rpm is a Kv-times-voltage ceiling rather
    #      than a knob to turn up. Prefer the overweight blade.
    #   3. Nothing reaches the target at all -> whatever lands closest, with
    #      meets_thrust False so the caller knows it fell short.
    #
    # Tier 2 matters more than it looks: with only five discrete blade counts
    # the achievable thrusts are quantized, so a candidate landing inside a
    # narrow band is partly luck. Without it, a tighter overshoot cap
    # paradoxically produces MORE undershoot, by rejecting the very candidates
    # that cleared the target.
    viable = [r for r in results if r["meets_thrust"]]
    over = [r for r in results if r["thrust_error_pct"] >= 0]
    if viable:
        best = max(viable, key=lambda r: r["thrust_per_watt_gf_W"])
    elif over:
        best = min(over, key=lambda r: r["thrust_error_pct"])
    else:
        best = min(results, key=lambda r: abs(r["thrust_error_pct"]))

    if blade_count is None:
        best = dict(best, candidates=results)
    return best


# --- command line -----------------------------------------------------------

def _fit(model_name):
    from dataset import load_synthetic
    Xs, ys, ps = load_synthetic()
    if model_name == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression().fit(Xs.values, ys.values)
    if model_name == "xgb":
        from models import XGBRaw
        return XGBRaw().fit(Xs.values, ys.values)
    raise SystemExit(f"unknown model: {model_name}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Design a hover propeller for a thrust requirement.",
        epilog="example: python src/design.py --thrust 500 --diameter 10 --rpm 6000",
    )
    ap.add_argument("--thrust", type=float, required=True,
                    help="target thrust in grams-force (use --newtons for N)")
    ap.add_argument("--newtons", action="store_true", help="treat --thrust as newtons")
    ap.add_argument("--diameter", type=float, required=True, help="diameter, inches")
    ap.add_argument("--rpm", type=float, required=True, help="target shaft speed")
    ap.add_argument("--blades", type=int, default=None,
                    help="fix the blade count; omit to auto-select from 2-6")
    ap.add_argument("--model", default="linear", choices=["linear", "xgb"],
                    help="linear is better calibrated on real propellers (default)")
    a = ap.parse_args()

    thrust_N = a.thrust if a.newtons else a.thrust * 9.80665 / 1000.0

    if not 2 <= a.diameter <= 30:
        print(f"note: diameter {a.diameter} in is outside SCOPE.md's 2-30 inch class")
    elif a.diameter < 6 or a.diameter > 11:
        print("note: Phase 3 validated efficiency within ~15% of the best real")
        print("      propeller only for 6-11 inch diameters. Outside that the model")
        print("      extrapolates below its training Reynolds range and did markedly")
        print("      worse -- treat the numbers below with caution.")

    model = _fit(a.model)
    r = design(model, thrust_N, a.diameter, a.rpm, a.blades)

    print(f"\nrequest : {thrust_N * GF_PER_N:.0f} gf at {a.diameter} in, {a.rpm:.0f} rpm")
    print(f"design  : {r['blade_count']} blades")
    print(f"\n  simulated thrust    {r['thrust_N'] * GF_PER_N:>8.1f} gf   "
          f"({r['thrust_error_pct']:+.1f}% vs request)")
    print(f"  shaft power         {r['shaft_power_W']:>8.1f} W")
    print(f"  thrust per watt     {r['thrust_per_watt_gf_W']:>8.2f} gf/W")
    print(f"  CT / CP             {r['CT']:>8.4f} / {r['CP']:.4f}")
    if not r["meets_thrust"]:
        if r["thrust_error_pct"] < 0:
            print(f"\n  WARNING: falls {abs(r['thrust_error_pct']):.0f}% short of the target.")
            print("           Target rpm is a motor ceiling, not a knob, so this would")
            print("           not meet the requirement. Try a larger diameter or rpm.")
        else:
            print(f"\n  NOTE: overshoots by {r['thrust_error_pct']:.0f}%. Meets the requirement")
            print("        but is oversized -- a smaller diameter or lower rpm fits closer.")

    if "candidates" in r:
        print("\n  blade-count candidates (* = chosen):")
        print(f"    {'':2}{'B':>3}{'thrust gf':>11}{'err %':>8}{'watts':>9}{'gf/W':>8}")
        for c in r["candidates"]:
            mark = "*" if c["blade_count"] == r["blade_count"] else " "
            print(f"    {mark:<2}{c['blade_count']:>3}{c['thrust_N'] * GF_PER_N:>11.1f}"
                  f"{c['thrust_error_pct']:>8.1f}{c['shaft_power_W']:>9.1f}"
                  f"{c['thrust_per_watt_gf_W']:>8.2f}")

    print(f"\n  geometry ({len(STATIONS)} stations)")
    print(f"    {'r/R':>6}{'c/R':>8}{'chord in':>10}{'twist deg':>11}")
    R_in = a.diameter / 2.0
    for x, c, b in zip(STATIONS, r["c_R"], r["beta_deg"]):
        print(f"    {x:>6.2f}{c:>8.4f}{c * R_in:>10.3f}{b:>11.2f}")


if __name__ == "__main__":
    main()
