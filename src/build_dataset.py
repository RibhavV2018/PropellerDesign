"""Assemble the Phase 1a dataset: one row per propeller operating point.

Joins three sources into a single flat table:

    filename  -> identity      (family, diameter, pitch, blade count)
    *_geom    -> LABELS        (chord and twist at 18 radial stations)
    *_static  -> FEATURES      (rpm, CT, CP at each measured operating point)

and derives the physical quantities the success metric is defined on
(thrust, shaft power, thrust per shaft watt).

WHY THERE IS NO pandas.merge HERE
Every row is built explicitly in a loop instead. A merge on prop_name would
be shorter, but a many-to-one merge silently multiplies rows when the right
side has duplicate keys, and silently drops them when a key is missing --
both of which this data actually has (repeat test runs; 8 propellers with no
parseable diameter). At 2000 rows an explicit loop costs nothing and makes
every inclusion and exclusion auditable, which is what the data dictionary
has to document. The assertions at the bottom are the real point.

    ./.venv/bin/python src/build_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_names import parse_prop_name
from physics import GF_PER_N, shaft_power_W, thrust_N, thrust_per_watt
from readers import find_geometry_files, find_static_runs, read_geometry, read_static

OUT = Path(__file__).resolve().parent.parent / "data" / "processed"

# Every usable propeller shares this station grid, verified below. Column
# names encode it (c_R_r015 = chord at r/R = 0.15) so the CSV is readable
# without the data dictionary.
GRID = np.round(np.arange(0.15, 1.001, 0.05), 2)


def geometry_columns(path) -> dict:
    """Flatten a geometry file into 36 label columns.

    Returns {c_R_r015: ..., beta_r015: ..., ...} -- chord and twist at each
    of the 18 stations. All 18 are kept rather than subsampled to SCOPE.md's
    6-10: that requirement describes what the TOOL outputs to a user, not
    what the dataset stores. Subsampling here would bake an irreversible
    modeling decision into the data layer, before Phase 2 has any evidence
    about whether raw stations, a 9-station subsample, or PCA scores predict
    better. Dropping columns later is easy; recovering them is not.
    """
    g = read_geometry(path)
    if not np.allclose(g.r_R.values, GRID):
        raise ValueError(f"{path}: unexpected station grid {g.r_R.tolist()}")
    cols = {}
    for r, c, b in zip(GRID, g.c_R, g.beta_deg):
        tag = f"r{int(round(r * 100)):03d}"
        cols[f"c_R_{tag}"] = c
        cols[f"beta_{tag}"] = b
    return cols


def build() -> tuple:
    """Build the dataset. Returns (DataFrame, exclusions dict)."""
    geom = find_geometry_files()
    runs = find_static_runs()

    excluded = {"no_geometry": set(), "unparsed_name": set()}
    records = []

    for run in runs:
        name = run["prop_name"]

        # Volumes 3-4 ship performance data with no geometry at all, and so
        # cannot supply labels no matter how good the performance data is.
        if name not in geom:
            excluded["no_geometry"].add(name)
            continue

        ident = parse_prop_name(name)
        # No diameter means no CT -> thrust conversion, so no target column.
        if not ident["parsed"]:
            excluded["unparsed_name"].add(name)
            continue

        labels = geometry_columns(geom[name])
        d_in = ident["diameter_in"]

        for row in read_static(run["path"]).itertuples():
            t = thrust_N(row.CT, row.rpm, d_in)
            p = shaft_power_W(row.CP, row.rpm, d_in)
            records.append({
                # identity / grouping keys
                "prop_name": name,
                "family": ident["family"],
                "blade_count": run["blade_count"],
                "diameter_in": d_in,
                "pitch_in": ident["pitch_in"],
                "source_file": run["path"].name,
                # measured operating point
                "rpm": row.rpm,
                "CT": row.CT,
                "CP": row.CP,
                # derived physical quantities
                "thrust_N": t,
                "thrust_gf": t * GF_PER_N,
                "shaft_power_W": p,
                "thrust_per_watt_N_W": thrust_per_watt(row.CT, row.CP, row.rpm, d_in),
                "thrust_per_watt_gf_W": thrust_per_watt(row.CT, row.CP, row.rpm, d_in) * GF_PER_N,
                **labels,
            })

    return pd.DataFrame(records), excluded


def verify(df: pd.DataFrame) -> None:
    """Assert the table is what we think it is.

    These catch the failure mode this script exists to avoid: rows silently
    appearing or disappearing during assembly.
    """
    label_cols = [c for c in df.columns if c.startswith(("c_R_", "beta_"))]
    assert len(label_cols) == 36, f"expected 36 label columns, got {len(label_cols)}"
    assert not df.isna().any().any(), "unexpected NaN in table"
    assert (df.thrust_N > 0).all(), "non-positive thrust"
    assert (df.shaft_power_W > 0).all(), "non-positive shaft power"
    assert df.diameter_in.between(2, 30).all(), "diameter outside SCOPE.md's 2-30 inch band"
    assert set(df.blade_count) <= {2, 3, 4}, "unexpected blade count"

    # Geometry must be identical across every row of the same propeller --
    # if a merge had duplicated or misaligned rows, this is where it shows.
    for name, grp in df.groupby("prop_name"):
        assert (grp[label_cols].nunique() == 1).all(), f"{name}: geometry varies within propeller"

    print("  all assertions passed")


def main() -> None:
    df, excluded = build()

    print(f"rows        : {len(df)}")
    print(f"propellers  : {df.prop_name.nunique()}")
    print(f"families    : {df.family.nunique()}")
    print(f"blade counts: {df.blade_count.value_counts().sort_index().to_dict()}")
    print(f"columns     : {len(df.columns)}")
    print()
    print(f"excluded -- no geometry file  : {len(excluded['no_geometry'])} propellers")
    print(f"excluded -- unparseable name  : {len(excluded['unparsed_name'])} propellers "
          f"{sorted(excluded['unparsed_name'])}")
    print()
    verify(df)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "uiuc_hover.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    print("\nranges:")
    for c in ["diameter_in", "rpm", "thrust_gf", "shaft_power_W", "thrust_per_watt_gf_W"]:
        print(f"  {c:<22} {df[c].min():10.2f} - {df[c].max():10.2f}   median {df[c].median():9.2f}")


if __name__ == "__main__":
    main()
