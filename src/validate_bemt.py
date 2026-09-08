"""Validate the BEMT solver against the 119 measured UIUC propellers.

Runs every geometry in data/processed/uiuc_hover.csv through the solver and
compares predicted CT and CP against wind-tunnel measurement. This is the
check that keeps the synthetic training data honest: if the solver is wrong,
a model trained on its output is confidently wrong in the same way, which is
the original paper's circular-ground-truth failure in better clothing.

WHAT TO READ IN THE OUTPUT

The headline medians are bias -- they respond to the airfoil model and should
be driven toward zero. The variance decomposition is the more important
number, and it does not:

    sd BETWEEN propellers  -- each propeller has its own unknown blade
                              section, and the solver assumes one airfoil for
                              all of them. No polar model can remove this;
                              the information is not in the database.
    sd WITHIN a propeller  -- how well the solver tracks one blade across its
                              own RPM sweep. This is the physics, and it
                              should be small.

A large within-propeller sd means the solver is wrong. A large
between-propeller sd with a small within means the solver is right and the
sections are unknown -- a floor to report, not a defect to fix.

    ./.venv/bin/python src/validate_bemt.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bemt import IN_M, MU, RHO, placeholder_airfoil, solve_propeller

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "uiuc_hover.csv"
OUT = ROOT / "data" / "processed" / "bemt_validation.csv"

STATIONS = [round(0.15 + 0.05 * k, 2) for k in range(18)]
C_COLS = [f"c_R_r{int(round(x * 100)):03d}" for x in STATIONS]
B_COLS = [f"beta_r{int(round(x * 100)):03d}" for x in STATIONS]


def reynolds_at_75(row) -> float:
    """Section Reynolds number at r/R = 0.75, the conventional reference.

    Uses the measured chord at that station and the hover section speed
    (omega*r; there is no freestream). Reported because the airfoil model's
    Reynolds dependence is the thing most likely to be wrong.
    """
    R = row.diameter_in * IN_M / 2.0
    U = 2.0 * np.pi * (row.rpm / 60.0) * 0.75 * R
    return RHO * U * (row.c_R_r075 * R) / MU


def run(airfoil=placeholder_airfoil, data=DATA) -> pd.DataFrame:
    """Solve every measured operating point. Returns measured vs predicted."""
    df = pd.read_csv(data)
    rows, failures = [], []

    for row in df.itertuples():
        try:
            out = solve_propeller(
                STATIONS,
                [getattr(row, c) for c in C_COLS],
                [getattr(row, b) for b in B_COLS],
                row.diameter_in, row.rpm, int(row.blade_count),
                airfoil=airfoil,
            )
        except Exception as exc:                  # recorded, never swallowed
            failures.append((row.prop_name, row.rpm, str(exc)))
            continue

        rows.append({
            "prop_name": row.prop_name, "family": row.family,
            "blade_count": row.blade_count, "diameter_in": row.diameter_in,
            "rpm": row.rpm, "Re_75": reynolds_at_75(row),
            "CT_measured": row.CT, "CP_measured": row.CP,
            "CT_solver": out["CT"], "CP_solver": out["CP"],
        })

    res = pd.DataFrame(rows)
    res["CT_error_pct"] = (res.CT_solver / res.CT_measured - 1) * 100
    res["CP_error_pct"] = (res.CP_solver / res.CP_measured - 1) * 100
    res.attrs["failures"] = failures
    res.attrs["n_input"] = len(df)
    return res


def report(res: pd.DataFrame) -> None:
    fails = res.attrs.get("failures", [])
    print(f"solved {len(res)} / {res.attrs['n_input']} operating points across "
          f"{res.prop_name.nunique()} propellers   ({len(fails)} failures)")
    for name, rpm, msg in fails[:5]:
        print(f"    FAILED  {name} @ {rpm:.0f} rpm: {msg}")

    print("\n--- bias (drive these toward zero with the airfoil model) ---")
    print(f"{'':14}{'median':>9}{'mean':>9}{'sd':>8}")
    for lab, col in [("CT error %", "CT_error_pct"), ("CP error %", "CP_error_pct")]:
        c = res[col]
        print(f"  {lab:<12}{c.median():>+9.1f}{c.mean():>+9.1f}{c.std():>8.1f}")

    print("\n--- variance decomposition (the number that does NOT move) ---")
    for lab, col in [("CT", "CT_error_pct"), ("CP", "CP_error_pct")]:
        between = res.groupby("prop_name")[col].median().std()
        within = res.groupby("prop_name")[col].std().median()
        print(f"  {lab}: between propellers {between:5.1f}%   "
              f"within one propeller {within:5.1f}%")
    print("  (between >> within  =>  physics is right, blade sections are unknown)")

    print("\n--- Reynolds trend (a slope here is missing Re physics) ---")
    bands = pd.cut(res.Re_75, [0, 25e3, 50e3, 75e3, np.inf],
                   labels=["<25k", "25-50k", "50-75k", ">75k"])
    tab = res.groupby(bands, observed=True).agg(
        n=("CT_error_pct", "size"),
        CT_err=("CT_error_pct", "median"),
        CP_err=("CP_error_pct", "median"),
    )
    print(tab.round(1).to_string())
    print(f"  corr(CT error, Re) = {res.CT_error_pct.corr(res.Re_75):+.3f}"
          f"     corr(CP error, Re) = {res.CP_error_pct.corr(res.Re_75):+.3f}")

    print("\n--- worst propellers by |median CT error| ---")
    worst = (res.groupby(["prop_name", "family"])
             .CT_error_pct.median().abs().sort_values(ascending=False).head(5))
    for (name, fam), err in worst.items():
        print(f"  {name:<22} ({fam:<8}) {err:6.1f}%")


def main() -> None:
    res = run()
    report(res)
    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
