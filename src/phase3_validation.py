"""Phase 3: closed-loop physics validation on held-out real propellers.

The question Phase 2's metrics do not answer: if you hand this tool a real
propeller's requirement, does the blade it designs actually do the job?

    1. take a real measured operating point -- thrust, diameter, rpm
    2. ask the tool to design a propeller for it
    3. simulate the designed blade in BEMT
    4. compare against the best REAL propeller meeting the same requirement

EVERY REAL PROPELLER IS HELD OUT

The models train on synthetic data only, so all 119 measured propellers are
unseen by construction. That is a stronger separation than a held-out split of
one dataset: the test set is not just different rows, it is different physics
provenance -- wind tunnel rather than solver.

WHY THE REFERENCE IS SIMULATED TOO

SCOPE.md compares against "the best available real reference propeller at that
same thrust/RPM/diameter". Both the designed blade and that reference are run
through the same solver. The solver's ~12% bias then cancels, so the
comparison measures design quality rather than solver error. Comparing a
simulated design against a measured reference would charge the tool for the
solver's bias.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bemt import low_re_airfoil, solve_propeller
from dataset import B_COLS, C_COLS, STATIONS, load_synthetic
from design import design
from physics import GF_PER_N

REAL = Path(__file__).resolve().parent.parent / "data" / "processed" / "uiuc_hover.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "phase3_validation.csv"

N_CASES = 15
CT_TOL = 0.15      # a reference must sit within 15% of the case's CT
D_TOL = 0.20       # ...and within 20% of its diameter


def simulate_real(row):
    """BEMT on a measured blade, so references and designs are comparable."""
    out = solve_propeller(STATIONS, [row[c] for c in C_COLS],
                          [row[b] for b in B_COLS],
                          row.diameter_in, row.rpm, int(row.blade_count),
                          airfoil=low_re_airfoil)
    return out


def pick_cases(df, n=N_CASES, seed=0):
    """Spread test cases across the diameter range rather than sampling at random.

    Random sampling would over-represent the small propellers that dominate
    the dataset, and the interesting failures are at the extremes.
    """
    rng = np.random.default_rng(seed)
    props = df.drop_duplicates("prop_name").sort_values("diameter_in")
    picks = props.iloc[np.linspace(0, len(props) - 1, n).round().astype(int)]
    cases = []
    for p in picks.itertuples():
        rows = df[df.prop_name == p.prop_name]
        cases.append(rows.iloc[len(rows) // 2])      # mid-RPM operating point
    return pd.DataFrame(cases)


def best_reference(df, case, sim_cache):
    """Best real propeller meeting the same requirement, by simulated CT/CP."""
    near = df[(np.abs(df.CT / case.CT - 1) <= CT_TOL)
              & (np.abs(df.diameter_in / case.diameter_in - 1) <= D_TOL)]
    near = near.drop_duplicates("prop_name")
    best, best_val = None, -np.inf
    # iterrows, not itertuples: simulate_real indexes columns by NAME, and a
    # namedtuple raises TypeError on a string key. That error has nothing to do
    # with solver convergence, so it must not be caught alongside one -- an
    # earlier version wrapped this in `except Exception` and silently produced
    # no reference for any case, which read as missing data rather than a bug.
    for _, r in near.iterrows():
        key = (r.prop_name, r.rpm)
        if key not in sim_cache:
            try:
                sim_cache[key] = simulate_real(r)
            except (ValueError, ZeroDivisionError):   # solver failures only
                sim_cache[key] = None
        o = sim_cache[key]
        if o and o["CP"] > 0:
            val = o["CT"] / o["CP"]
            if val > best_val:
                best, best_val = r, val
    return best, sim_cache.get((best.prop_name, best.rpm)) if best is not None else None


def main() -> None:
    df = pd.read_csv(REAL)
    Xs, ys, _ = load_synthetic()
    model = LinearRegression().fit(Xs.values, ys.values)

    cases = pick_cases(df)
    sim_cache = {}
    rows = []

    for case in cases.itertuples():
        d = design(model, case.thrust_N, case.diameter_in, case.rpm)  # auto blade count
        ref, ref_sim = best_reference(df, case, sim_cache)
        ref_gfw = (ref_sim["thrust_N"] * GF_PER_N / ref_sim["shaft_power_W"]
                   if ref_sim else np.nan)
        rows.append({
            "case": case.prop_name,
            "D_in": case.diameter_in,
            "rpm": case.rpm,
            "target_gf": case.thrust_N * GF_PER_N,
            "got_gf": d["thrust_N"] * GF_PER_N,
            "thrust_err_pct": d["thrust_error_pct"],
            "B_chosen": d["blade_count"],
            "B_ref": int(ref.blade_count) if ref is not None else np.nan,
            "design_gfW": d["thrust_per_watt_gf_W"],
            "ref_gfW": ref_gfw,
            "ref_prop": ref.prop_name if ref is not None else "-",
            "vs_ref_pct": (d["thrust_per_watt_gf_W"] / ref_gfw - 1) * 100 if ref_sim else np.nan,
            "meets": d["meets_thrust"],
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    print("--- Phase 3: designed propeller vs best real reference ---")
    print("    (both simulated in the same solver, so solver bias cancels)\n")
    show = res[["case", "D_in", "rpm", "target_gf", "got_gf", "thrust_err_pct",
                "B_chosen", "B_ref", "design_gfW", "ref_gfW", "vs_ref_pct"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    ok = res.dropna(subset=["vs_ref_pct"])
    print(f"\n  cases meeting thrust within 15%   : {res.meets.sum()} / {len(res)}")
    print(f"  median |thrust error|              : {res.thrust_err_pct.abs().median():.1f}%")
    print(f"  median efficiency vs best reference: {ok.vs_ref_pct.median():+.1f}%")
    print(f"  cases at or above the reference    : {(ok.vs_ref_pct >= 0).sum()} / {len(ok)}")
    match = ok[ok.B_chosen == ok.B_ref]
    print(f"  auto-selected blade count == reference: {len(match)} / {len(ok)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
