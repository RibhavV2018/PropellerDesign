"""Pre-generate example designs for the static showcase site (docs/).

    ./.venv/bin/python src/export_showcase.py

WHY THIS EXISTS

The Flask app can't be hosted for free: CadQuery and VTK are about 2 GB of
dependencies, and generating one propeller peaks near 760 MB of memory. So the
site on GitHub Pages is static. This script runs the same pipeline app.py runs
(design -> build_propeller -> export -> render) for a handful of requirements
and writes everything the page needs to docs/designs/<name>/.

FORCED VS AUTO-SELECTED BLADE COUNTS

design() only returns its candidate table when it chooses the blade count
itself. Some showcase designs force a count (auto-select never picks 4 blades
anywhere in a 5-15 inch sweep), so every design is also run once in auto mode
on the same requirement. That records what auto-select WOULD have chosen,
which the page states rather than hides.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import get_model, warnings_for
from cad import build_propeller, check_manifold, export, render
from dataset import STATIONS
from design import design
from physics import GF_PER_N

import cadquery as cq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "designs"
PHASE3 = ROOT / "data" / "processed" / "phase3_validation.csv"

G = 9.80665 / 1000.0          # grams-force -> newtons
PLA_G_CM3 = 1.24
BORE_MM = 6.0

# An STL above this is re-exported with a coarser tessellation. An old 6 in
# 4-blade export came out at 25 MB -- far too large to serve from Pages.
MAX_STL_BYTES = 3_000_000
COARSE = dict(tolerance=0.05, angularTolerance=0.3)   # mm, radians

# (name, diameter in, rpm, target gf, forced blade count or None, blurb)
DESIGNS = [
    ("6in-4b", 6, 12000, 500, 4,
     "Small quad prop. Four blades are forced here: auto-select prefers three."),
    ("8in-3b", 8, 6000, 600, 3,
     "Mid-size multirotor prop. Auto-select independently picks three blades."),
    ("10in-2b", 10, 7500, 900, 2,
     "Classic two-blade layout, landing almost exactly on the thrust target."),
    ("12in-5b", 12, 5000, 500, 5,
     "Lightly loaded, so auto-select reaches for five blades. Outside the "
     "validated 6-11 in range."),
    ("15in-5b", 15, 4000, 800, 5,
     "Large, slow prop for heavy-lift hover. Outside the validated range."),
    ("8in-oversized", 8, 10000, 400, None,
     "A requirement no blade count fits: the best candidate overshoots "
     "thrust by about 23%, and the tool says so."),
]


def candidate_rows(result):
    return [{"blades": c["blade_count"],
             "thrust_gf": c["thrust_N"] * GF_PER_N,
             "error_pct": c["thrust_error_pct"],
             "shaft_power_W": c["shaft_power_W"],
             "gf_per_W": c["thrust_per_watt_gf_W"],
             "meets": c["meets_thrust"]}
            for c in result["candidates"]]


def stl_to_glb(stl_path, glb_path):
    """Convert for <model-viewer>. glTF is Y-up and in metres; CAD is Z-up, mm."""
    import numpy as np
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    mesh.apply_scale(0.001)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.merge_vertices()
    mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[122, 166, 214, 255])
    mesh.export(glb_path)


def export_one(model, name, D, rpm, target_gf, blades, blurb):
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    r = design(model, target_gf * G, D, rpm, blades)
    auto = design(model, target_gf * G, D, rpm, None)

    prop = build_propeller(r["c_R"], r["beta_deg"], D, r["blade_count"], bore_mm=BORE_MM)
    paths = export(prop, name, out_dir=out)
    stl = next(p for p in paths if p.suffix == ".stl")
    if stl.stat().st_size > MAX_STL_BYTES:
        print(f"  {stl.stat().st_size / 1e6:.1f} MB STL -> re-exporting coarser")
        cq.exporters.export(prop, str(stl), **COARSE)
    png = render(stl)
    mesh = check_manifold(stl)
    glb = out / f"{name}.glb"
    stl_to_glb(stl, glb)

    vol_cm3 = prop.val().Volume() / 1000.0
    R_in = D / 2.0
    data = {
        "name": name,
        "blurb": blurb,
        "inputs": {"diameter_in": D, "rpm": rpm, "thrust_target_gf": target_gf,
                   "blades_requested": blades, "bore_mm": BORE_MM},
        "blade_count": r["blade_count"],
        "blade_count_forced": blades is not None,
        "auto_blade_count": auto["blade_count"],
        "thrust_gf": r["thrust_N"] * GF_PER_N,
        "thrust_error_pct": r["thrust_error_pct"],
        "meets_thrust": r["meets_thrust"],
        "shaft_power_W": r["shaft_power_W"],
        "gf_per_W": r["thrust_per_watt_gf_W"],
        "CT": r["CT"], "CP": r["CP"],
        "volume_cm3": vol_cm3,
        "mass_pla_g": vol_cm3 * PLA_G_CM3,
        "triangles": mesh["triangles"],
        "watertight": mesh["watertight"],
        "geometry": [{"r_R": x, "c_R": c, "chord_in": c * R_in, "twist_deg": b}
                     for x, c, b in zip(STATIONS, r["c_R"], r["beta_deg"])],
        "candidates": candidate_rows(auto),
        "warnings": warnings_for(D, r),
        "files": {k: p.name for k, p in
                  [("stl", stl), ("step", out / f"{name}.step"), ("png", png), ("glb", glb)]},
    }
    (out / "design.json").write_text(json.dumps(data, indent=2))
    return data


def main() -> None:
    print("fitting model...")
    model = get_model()
    index = []
    for name, D, rpm, target, blades, blurb in DESIGNS:
        print(f"{name}: {D} in, {rpm} rpm, {target} gf, blades={blades or 'auto'}")
        d = export_one(model, name, D, rpm, target, blades, blurb)
        sizes = {k: (OUT / name / f).stat().st_size / 1e6 for k, f in d["files"].items()}
        print(f"  -> {d['blade_count']} blades, {d['thrust_gf']:.0f} gf "
              f"({d['thrust_error_pct']:+.1f}%), {d['gf_per_W']:.2f} gf/W, "
              f"watertight={d['watertight']}, "
              + ", ".join(f"{k} {v:.2f} MB" for k, v in sizes.items()))
        index.append(name)
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    # The validation table on the page reads this copy, so rerunning here
    # keeps the site in step with the latest Phase 3 run.
    (ROOT / "docs" / "data").mkdir(exist_ok=True)
    shutil.copy(PHASE3, ROOT / "docs" / "data" / PHASE3.name)
    print(f"wrote {len(index)} designs to {OUT}")


if __name__ == "__main__":
    main()
