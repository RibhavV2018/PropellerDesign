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


def find_static_files(root: Path = DATA_ROOT) -> dict:
    """Map prop_name -> [Path, ...] for every static performance file.

    A list, not a single path: propellers are often tested more than once
    (different rigs, dates, or operators -- the kt/rd/os/jb tags in the
    filename), so one propeller can have several static files with
    overlapping RPM ranges. Deciding what to do with duplicate runs is a
    join-time question.
    """
    out = {}
    for p in sorted(root.glob("volume-*/data/*_static_*.txt")):
        name = p.name.split("_static_")[0]
        out.setdefault(name, []).append(p)
    return out


def main() -> None:
    geom = find_geometry_files()
    static = find_static_files()

    print(f"geometry files : {len(geom)} propellers")
    print(f"static files   : {sum(len(v) for v in static.values())} files "
          f"across {len(static)} propellers")

    both = sorted(set(geom) & set(static))
    print(f"propellers with BOTH (usable): {len(both)}")

    multi = {k: len(v) for k, v in static.items() if k in both and len(v) > 1}
    print(f"usable propellers with >1 static run: {len(multi)}")
    if multi:
        top = sorted(multi.items(), key=lambda kv: -kv[1])[:5]
        print(f"  most-retested: {top}")

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
