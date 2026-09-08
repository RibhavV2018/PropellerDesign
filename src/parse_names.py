"""Parse UIUC propeller filenames into structured identity fields.

The UIUC data files encode the propeller's identity in the FILENAME, not in the
file contents. Diameter in particular appears nowhere inside the data files --
they hold only coefficients (CT, CP) -- so without a correct diameter here,
nothing downstream converts to Newtons and watts.

Naming convention (the common case):

    apcsf_9x4.7_geom.txt            -> family=apcsf, D=9.0 in, pitch=4.7 in
    apcsf_9x4.7_static_kt1032.txt   -> same prop, static (hover) performance
    apcsf_9x4.7_kt1033_4008.txt     -> same prop, dynamic (J-sweep) -- NOT v1

Run the checks at the bottom with:

    ./.venv/bin/python src/parse_names.py
"""

import re

# Propellers whose filename dimensions are in MILLIMETERS, not inches.
# There is no way to tell from the name alone -- verified by computing the
# thrust each interpretation implies (inches gives 12-45 tonnes on a wind
# tunnel bench; mm gives a sane 29-108 gram-force).
MM_PROPS = {
    "vp_140x45",
    "pl_100x80",
    "kpf_96x70",
    "ef_130x70",
    "pl_57x20",
}

MM_TO_IN = 0.0393701

# _(digits, optional decimal) x (digits, optional decimal)
# The leading underscore is defensive rather than load-bearing: it makes no
# difference on the current 127-prop corpus, but it states that the size
# follows the family prefix, and 12 families have digits in their names.
SIZE_RE = re.compile(r"_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)")


def parse_prop_name(stem: str) -> dict:
    """Parse a propeller identity string into structured fields.

    Args:
        stem: the propeller identity, i.e. the filename with its type suffix
            already stripped -- "apcsf_9x4.7", not "apcsf_9x4.7_geom.txt".

    Returns:
        dict with keys:
            prop_name    -- the stem, unchanged (your grouping key for CV)
            family       -- manufacturer/series prefix, e.g. "apcsf"
            diameter_in  -- diameter in INCHES, or None if not encoded
            pitch_in     -- pitch in INCHES, or None if not encoded
            parsed       -- bool, whether size was recoverable from the name

    Four propellers in the corpus encode no size at all (cfnq_45p1, pl_triturbo,
    union_u80, nr640_5_15deg -- the last states a blade angle, not a length).
    They come back with parsed=False and None sizes; the caller decides whether
    to drop them. Note this in the data dictionary so the drop is visible.
    """
    out = {
        "prop_name": stem,
        "family": stem.split("_")[0],
        "diameter_in": None,
        "pitch_in": None,
        "parsed": False,
    }

    m = SIZE_RE.search(stem)
    if m is None:
        return out

    diameter = float(m.group(1))
    pitch = float(m.group(2))

    if stem in MM_PROPS:
        diameter *= MM_TO_IN
        pitch *= MM_TO_IN

    out["diameter_in"] = diameter
    out["pitch_in"] = pitch
    out["parsed"] = True
    return out


# --- checks -----------------------------------------------------------------
# Real cases pulled from the downloaded corpus: the clean majority first,
# then every irregular name that actually exists in the 127 usable props.

CASES = [
    # (stem, expected diameter_in, expected pitch_in)
    ("apcsf_9x4.7", 9.0, 4.7),        # the common case
    ("apce_11x5.5", 11.0, 5.5),
    ("gwsdd_2.5x0.8", 2.5, 0.8),      # decimals on both sides
    ("apcff_4.2x4", 4.2, 4.0),        # decimal on one side only
    ("gwsdd_5x4.3_spec2", 5.0, 4.3),  # trailing variant tag
    ("vp_140x45", 5.512, 1.772),      # MILLIMETERS -> inches
    ("pl_57x20", 2.244, 0.787),       # MILLIMETERS -> inches
    ("cfnq_45p1", None, None),        # no size encoded
    ("nr640_5_15deg", None, None),    # pitch is a blade angle, not a length
    ("pl_triturbo", None, None),      # no size encoded
    ("union_u80", None, None),        # no size encoded
]


def main() -> None:
    passed = failed = 0
    for stem, want_d, want_p in CASES:
        got = parse_prop_name(stem)
        ok_d = _close(got.get("diameter_in"), want_d)
        ok_p = _close(got.get("pitch_in"), want_p)
        if ok_d and ok_p:
            print(f"  PASS  {stem:<22} D={got.get('diameter_in')} P={got.get('pitch_in')}")
            passed += 1
        else:
            print(f"  FAIL  {stem:<22} got D={got.get('diameter_in')} P={got.get('pitch_in')}"
                  f"  want D={want_d} P={want_p}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")


def _close(got, want, tol=0.01) -> bool:
    if want is None or got is None:
        return got is want or got == want
    return abs(got - want) < tol


if __name__ == "__main__":
    main()
