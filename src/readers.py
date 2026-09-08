"""Readers for the two UIUC file formats used in v1.

The database ships three kinds of data file. Only two matter for a
hover-optimized design:

    *_geom.txt          blade geometry -- the LABELS the model predicts
    *_static_*.txt      static (J=0) performance -- hover, the DESIGN POINT
    *_<run>_<rpm>.txt   advance-ratio sweeps -- forward flight, i.e. the
                        cruise case, which SCOPE.md defers to a stretch
                        extension. Not read here.

Both formats are whitespace-delimited with a single header line and exactly
three columns. Header spacing varies between volumes, so column names are
assigned positionally rather than parsed from the file.

Note that geometry exists only in volumes 1-2; volumes 3-4 ship performance
data with no geometry at all, so they cannot supply training labels. The
discovery helpers below still scan all four volumes -- the mismatch is real
and belongs in the join, not hidden here.

    ./.venv/bin/python src/readers.py
"""

import re
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "UIUC-propDB"


def read_geometry(path) -> pd.DataFrame:
    """Read a *_geom.txt file.

    Returns a DataFrame with columns:
        r_R      -- radial station as a fraction of tip radius (0.15 .. 1.0)
        c_R      -- chord at that station, normalized by tip radius R
        beta_deg -- blade angle (twist) at that station, in degrees

    Both c_R and r_R are normalized by R, not by diameter. To get physical
    chord in inches: c_R * (diameter_in / 2).

    Files carry 15 or 18 stations depending on the test; resampling to a
    fixed station count is a downstream concern, not done here.
    """
    return pd.read_csv(
        path, sep=r"\s+", skiprows=1, names=["r_R", "c_R", "beta_deg"], dtype=float
    )


def read_static(path) -> pd.DataFrame:
    """Read a *_static_*.txt file (hover / J=0 performance).

    Returns a DataFrame with columns:
        rpm  -- shaft speed
        CT   -- thrust coefficient,  T = CT * rho * n^2 * D^4   (n in rev/s)
        CP   -- power coefficient,   P = CP * rho * n^3 * D^5

    Both coefficients are dimensionless and use the propeller convention
    (normalized by n and D, not by tip speed), so D must come from the
    filename -- see parse_names.parse_prop_name.
    """
    return pd.read_csv(
        path, sep=r"\s+", skiprows=1, names=["rpm", "CT", "CP"], dtype=float
    )


def find_geometry_files(root: Path = DATA_ROOT) -> dict:
    """Map prop_name -> Path for every geometry file.

    Mostly 1:1, with one exception: gwsdd_9x5 has a geometry file in BOTH
    volume-1 and volume-2. The two are numerically identical (they differ
    only in trailing whitespace), so the first is kept.

    A silent dict overwrite would also "work" here, but only by luck -- it
    would hide the day two volumes disagree about the same blade. So the
    duplicate is checked rather than assumed, and a genuine conflict raises.
    """
    out = {}
    for p in sorted(root.glob("volume-*/data/*_geom.txt")):
        name = p.name[: -len("_geom.txt")]
        prev = out.get(name)
        if prev is None:
            out[name] = p
        elif not read_geometry(prev).equals(read_geometry(p)):
            raise ValueError(
                f"{name}: geometry differs between {prev} and {p}. "
                "Decide which test to trust before building the table."
            )
    return out


def parse_static_filename(path: Path):
    """Split a static filename into (base propeller name, blade count).

    Blade count is encoded as a _3b_ / _4b_ infix before "_static_":

        da4022_5x3.75_static_0642rb.txt      -> ("da4022_5x3.75", 2)
        da4022_5x3.75_3b_static_0690md.txt   -> ("da4022_5x3.75", 3)

    Two-blade is the unmarked default -- 253 of 264 static files. The base
    name matters because the multi-blade variants have NO geometry file of
    their own: they reuse the base propeller's blade, changing only how many
    of them are on the hub. Keying on the raw stem instead of the base name
    orphans all 11 multi-blade files from their geometry.
    """
    stem = path.name.split("_static_")[0]
    m = re.search(r"_(\d)b$", stem)
    if m:
        return stem[: m.start()], int(m.group(1))
    return stem, 2


def find_static_runs(root: Path = DATA_ROOT) -> list:
    """Every static performance file, as a list of records.

    Each record is a dict with:
        prop_name    -- BASE propeller name, matching its geometry file
        blade_count  -- 2, 3 or 4, from the filename infix
        path         -- Path to the data file

    A list rather than a dict because one propeller can contribute several
    rows: different blade counts, and repeat test runs of the same
    configuration (apcsp_9x6 and gwsdd_9x5 were each tested twice). Both are
    legitimately separate operating data for the same blade, and both must
    stay on the same side of any train/test split -- group on prop_name.
    """
    runs = []
    for p in sorted(root.glob("volume-*/data/*_static_*.txt")):
        name, blades = parse_static_filename(p)
        runs.append({"prop_name": name, "blade_count": blades, "path": p})
    return runs


def main() -> None:
    from collections import Counter

    geom = find_geometry_files()
    runs = find_static_runs()

    print(f"geometry files : {len(geom)} propellers")
    print(f"static runs    : {len(runs)} files")
    print(f"  by blade count: {dict(sorted(Counter(r['blade_count'] for r in runs).items()))}")

    named = {r["prop_name"] for r in runs}
    both = sorted(set(geom) & named)
    print(f"propellers with BOTH geometry and static: {len(both)}")
    print(f"static runs usable (base prop has geometry): "
          f"{sum(1 for r in runs if r['prop_name'] in geom)}")

    static = {}
    for r in runs:
        static.setdefault(r["prop_name"], []).append(r["path"])
    ex = both[0]
    print(f"\n--- example: {ex} ---")
    print("geometry (first 4 stations):")
    print(read_geometry(geom[ex]).head(4).to_string(index=False))
    print("\nstatic (first 4 rows):")
    print(read_static(static[ex][0]).head(4).to_string(index=False))

    rows = sum(len(read_static(p)) for k in both for p in static[k])
    print(f"\ntotal static operating points across usable props: {rows}")


if __name__ == "__main__":
    main()
