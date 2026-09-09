"""Generate synthetic hover propellers by sampling geometry and running BEMT.

Phase 1b deliverable. The real UIUC data has two limits this fixes:

  n = 119 independent blades, for a model with 36 outputs.
  Blade count is effectively a constant -- 2000 rows at 2 blades, 121 at 3-4,
  from four base propellers. SCOPE.md's auto-select-best-blade-count feature
  cannot be learned from that. Here blade count is swept 2-6 by construction.

WHERE THE ASSUMPTION LIVES

Geometry is sampled from a low-dimensional physical parameterization, not
per-station at random (which gives jagged, unmanufacturable blades) and not by
perturbing real ones (which barely widens the design space). But the
PERFORMANCE LABELS still come from running each sampled blade through BEMT.

That distinction is the whole point. The original AIAA paper baked a fixed
assumption -- constant efficiency, constant angle of attack -- into the
LABELS, so a model trained on them just relearned the assumption. Here the
assumption is in the sampler, and the labels come from physics.

TWIST -- one parameter, plus washout

    beta(r) = arctan(P / (2*pi*r))  +  washout * (r/R - 0.75)

Constant geometric pitch. This reproduces real blades to a median 2.05 deg
using nothing but nameplate pitch, and it is better than that for the
hover-oriented families (apcsf 1.16 deg) which are the primary case.

It is also close to the aerodynamic optimum, not just the manufacturing
convention: ideal hover twist for uniform inflow is theta(r) = theta_tip*R/r,
and constant geometric pitch is the same 1/r law in the small-angle limit --
the two agree within 0.69 deg over r/R 0.3-1.0. The washout term adds the
tip-stall margin real designers apply on top.

CHORD -- magnitude times shape

Real blades separate cleanly: a magnitude (solidity) and a normalized shape
needing only 2-3 numbers (82% / 90% of shape variance). Mid-span shape is
nearly universal; the variation is at root and tip. Shape components are the
principal components of the 119 measured blades.

RPM IS NOT SAMPLED DIRECTLY

Tip speed is the physically bounded quantity; rpm is not. In the real data rpm
correlates -0.54 with diameter because small propellers spin fast. Sampling
rpm independently produces a 30 inch propeller at 20,000 rpm -- tip Mach 2.33,
in a solver with no compressibility model. Those samples would be silently and
confidently wrong. So tip speed is sampled and rpm derived:

    rpm = 60 * V_tip / (pi * D)

    ./.venv/bin/python src/generate_synthetic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bemt import IN_M, MU, RHO, low_re_airfoil, solve_propeller
from validate_bemt import B_COLS, C_COLS, DATA, STATIONS

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT = PROC / "synthetic_hover.csv"          # every sample
OUT_FRONTIER = PROC / "synthetic_frontier.csv"   # the efficient subset, for training

FRONTIER_PCT = 0.15   # keep the top this fraction of each bin

N_SAMPLES = 200_000
SEED = 0

# Ranges are modestly wider than the measured data, so the model interpolates
# rather than extrapolates at the edges of real design space. Measured spans
# are given for comparison.
DIAMETER_IN = (2.0, 30.0)      # SCOPE.md's class; data covers 2.2-19
TIP_SPEED_MS = (15.0, 100.0)   # data 17-83; keeps tip Mach under 0.3
PITCH_RATIO = (0.25, 1.00)     # P/D; data 0.32-0.91
SOLIDITY = (0.05, 0.22)        # data 0.065-0.186
WASHOUT_DEG = (-4.0, 2.0)      # deviation from pure constant pitch
BLADE_COUNTS = (2, 3, 4, 5, 6)  # SCOPE.md's auto-select candidate set
# Chord-shape sampling width, as a MULTIPLE of the spread real propellers
# actually show. Sampling wide seemed generous and was not: at the original
# setting the synthetic coefficient spread was 2.5-3.1x the real one and 70%
# of generated blades fell outside the real shape envelope. A model trained on
# them learned a chord distribution no manufacturer builds, which fitted
# synthetic data well (chord R2 0.739) and transferred badly to real
# propellers (-3.511). 1.5 still extrapolates beyond observed practice without
# spending most of the samples outside it.
SHAPE_WIDEN = 1.5

_ST = np.array(STATIONS)


def chord_shape_basis(data=DATA):
    """Mean and principal components of normalized chord shape, from real blades.

    Chord is split into magnitude and shape: each measured blade is divided by
    its own peak, so what remains is planform shape alone. Two components
    cover 82% of the variation in that shape, three cover 90%.

    Returns:
        (mean_shape, components, coef_sd) -- components shaped (2, 18), and
        the standard deviation real blades show along each component, which
        is what the sampler scales by so synthetic shapes stay in the range
        real propellers occupy.
    """
    df = pd.read_csv(data).drop_duplicates("prop_name")
    C = df[C_COLS].values
    shape = C / C.max(axis=1, keepdims=True)
    mean = shape.mean(axis=0)
    _, _, vt = np.linalg.svd(shape - mean, full_matrices=False)
    comps = vt[:2]
    coef_sd = ((shape - mean) @ comps.T).std(axis=0)
    return mean, comps, coef_sd


def make_geometry(pitch_ratio, diameter_in, solidity, washout_deg,
                  shape_coeffs, blade_count, mean_shape, components):
    """Build one blade: 18 chord values (c/R) and 18 twist values (degrees)."""
    R_in = diameter_in / 2.0
    r_in = _ST * R_in

    # Twist: constant geometric pitch, plus linear washout about r/R = 0.75.
    P_in = pitch_ratio * diameter_in
    beta = np.degrees(np.arctan(P_in / (2.0 * np.pi * r_in)))
    beta = beta + washout_deg * (_ST - 0.75)

    # Chord: normalized shape from the PCA basis, scaled to hit the target
    # solidity. Clipped positive -- a sampled shape can dip below zero at the
    # tip, which is not a blade.
    shape = mean_shape + shape_coeffs @ components
    shape = np.clip(shape, 0.01, None)
    # solidity = B * integral(c/R) d(r/R) / pi  -> solve for the scale factor
    integral = np.trapezoid(shape, _ST)
    scale = solidity * np.pi / (blade_count * integral)
    c_R = shape * scale

    return c_R, beta



def efficient_frontier(df, pct=FRONTIER_PCT):
    """Keep the most efficient blades within each requirement bin.

    WHY FILTER AT ALL

    Many different blades produce the same CT, and they are not equally good.
    All satisfy the user's stated requirement; they differ in what that thrust
    costs. A regressor trained on all of them minimizes squared error and so
    predicts the CONDITIONAL MEAN geometry, when what an inverse design tool
    owes the user is the conditional OPTIMUM.

    HOW BIG IS THE EFFECT, HONESTLY

    Measured on bins that control for Reynolds number, the best blade beats
    the median by 1.15x in CT/CP, and filtering to the top 15% lifts median
    CT/CP from 1.93 to 2.27 -- about 18%. Real, worth having, not dramatic.

    An earlier estimate of 3.3x was wrong. It binned on (CT, blade count,
    diameter class) and ranked on thrust per watt, which leaves the 1/(n*D)
    factor inside the bin: the "best" blades were simply the largest and
    slowest, not the aerodynamically better ones. Ranking a dimensionless
    quantity on Reynolds-controlled bins removes that confound. The lesson
    generalizes -- a spread measured on the wrong metric can be almost
    entirely an artifact of what the bin failed to hold constant.

    WHY CT/CP, NOT THRUST PER WATT

        T/P = (CT * rho * n^2 * D^4) / (CP * rho * n^3 * D^5) = (CT/CP) / (n*D)

    Thrust per watt carries a 1/(n*D) factor, so ranking on it inside a loose
    bin just selects the largest, slowest propellers -- a trivial result that
    says nothing about blade quality. CT/CP is dimensionless and isolates the
    aerodynamics. At fixed CT it is equivalent to minimizing CP, which is
    exactly what the user wants: the same thrust for less power.

    Bins control for Reynolds number as well, since achievable CT/CP falls at
    low Re; without that the frontier would be all large high-Re blades.
    """
    d = df[df.plausible].copy()
    d["ct_cp"] = d.CT / d.CP
    d["_ct"] = pd.cut(d.CT, 20)
    d["_re"] = pd.qcut(d.Re_75, 4, duplicates="drop")

    keep = (d.groupby(["_ct", "blade_count", "_re"], observed=True, group_keys=False)
              .apply(lambda g: g.nlargest(max(1, int(round(len(g) * pct))), "ct_cp")))
    return keep.drop(columns=["_ct", "_re"])


def main() -> None:
    rng = np.random.default_rng(SEED)
    mean_shape, components, coef_sd = chord_shape_basis()

    rows, failures = [], 0
    for i in range(N_SAMPLES):
        D = rng.uniform(*DIAMETER_IN)
        v_tip = rng.uniform(*TIP_SPEED_MS)
        pd_ratio = rng.uniform(*PITCH_RATIO)
        solidity = rng.uniform(*SOLIDITY)
        washout = rng.uniform(*WASHOUT_DEG)
        B = int(rng.choice(BLADE_COUNTS))
        coeffs = rng.normal(0.0, 1.0, size=2) * coef_sd * SHAPE_WIDEN

        rpm = 60.0 * v_tip / (np.pi * D * IN_M)
        c_R, beta = make_geometry(pd_ratio, D, solidity, washout, coeffs, B,
                                  mean_shape, components)

        try:
            out = solve_propeller(_ST.tolist(), c_R.tolist(), beta.tolist(),
                                  D, rpm, B, airfoil=low_re_airfoil)
        except Exception:
            failures += 1
            continue

        rec = {
            "sample_id": i,
            "blade_count": B, "diameter_in": D, "pitch_in": pd_ratio * D,
            "pitch_ratio": pd_ratio, "solidity": solidity,
            "washout_deg": washout, "shape_c1": coeffs[0], "shape_c2": coeffs[1],
            "tip_speed_ms": v_tip, "rpm": rpm,
            "Re_75": RHO * (2 * np.pi * (rpm / 60.0) * 0.75 * (D * IN_M / 2))
                     * (c_R[12] * D * IN_M / 2) / MU,
            "CT": out["CT"], "CP": out["CP"],
            "thrust_N": out["thrust_N"], "shaft_power_W": out["shaft_power_W"],
            "thrust_per_watt_N_W": out["thrust_N"] / out["shaft_power_W"]
            if out["shaft_power_W"] > 0 else np.nan,
        }
        for x, c, b in zip(STATIONS, c_R, beta):
            tag = f"r{int(round(x * 100)):03d}"
            rec[f"c_R_{tag}"] = c
            rec[f"beta_{tag}"] = b
        rows.append(rec)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,} / {N_SAMPLES:,}")

    df = pd.DataFrame(rows)

    # Sampling the ranges independently produces some physically valid but
    # useless propellers -- a 30 inch blade at 15 m/s tip speed barely makes
    # thrust. They are FLAGGED, not dropped, so Phase 2 decides rather than
    # having the choice baked into the data. Same principle as storing all 18
    # geometry stations.
    df["plausible"] = (
        (df.thrust_N > 0.05)
        & (df.CT > 0.01)
        & (df.CP > 0.001)
        & df.thrust_per_watt_N_W.notna()
    )

    df.to_csv(OUT, index=False)
    print(f"\ngenerated {len(df):,} propellers ({failures} solver failures)")
    print(f"plausible: {df.plausible.sum():,} ({df.plausible.mean()*100:.1f}%)")
    print(f"\nblade counts: {df[df.plausible].blade_count.value_counts().sort_index().to_dict()}")
    p = df[df.plausible]
    print("\nranges (plausible only):")
    for c in ["diameter_in", "rpm", "thrust_N", "shaft_power_W", "CT", "CP"]:
        print(f"  {c:<18}{p[c].min():>11.3f} - {p[c].max():>11.1f}   median {p[c].median():>9.3f}")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")

    front = efficient_frontier(df)
    front.to_csv(OUT_FRONTIER, index=False)
    print(f"\n--- efficient frontier (top {FRONTIER_PCT:.0%} of each bin) ---")
    print(f"  {len(front):,} blades   blade counts "
          f"{front.blade_count.value_counts().sort_index().to_dict()}")
    print(f"  CT/CP  frontier median {(front.CT/front.CP).median():.2f}   "
          f"all-samples median {(p.CT/p.CP).median():.2f}")
    fg = front.thrust_per_watt_N_W * 1000/9.80665
    ag = p.thrust_per_watt_N_W * 1000/9.80665
    print(f"  gf/W   frontier median {fg.median():.1f}   all-samples median {ag.median():.1f}")
    print(f"  wrote {OUT_FRONTIER}  ({OUT_FRONTIER.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
