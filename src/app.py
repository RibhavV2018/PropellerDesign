"""Web frontend for the propeller designer (Phase 6).

    ./.venv/bin/python src/app.py     then open http://127.0.0.1:5000

TWO ENDPOINTS, BECAUSE THEY HAVE VERY DIFFERENT COSTS

    /api/design   ~0.1 s   model inference plus a handful of BEMT solves
    /api/cad      ~3 s     lofting and boolean unions in the CAD kernel

Splitting them keeps the form responsive: you can iterate on requirements and
see performance immediately, and only pay for geometry when you actually want
a file. Fitting the model happens once at startup, not per request.

The page states its own limits rather than presenting every number as equally
trustworthy. Phase 3 validated efficiency to within about 15% of the best real
propeller for 6-11 inch diameters; outside that the model extrapolates below
its training Reynolds range and did markedly worse. Performance figures are
SIMULATED, never measured.
"""

import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cad import build_propeller, check_manifold, export, render
from dataset import STATIONS, load_synthetic
from design import CANDIDATE_BLADES, design
from physics import GF_PER_N

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "web"

app = Flask(__name__, template_folder=str(ROOT / "templates"),
            static_folder=str(ROOT / "static"))

_model = None


def get_model():
    """Fit once, on first use -- takes a few seconds and never changes."""
    global _model
    if _model is None:
        from sklearn.linear_model import LinearRegression
        Xs, ys, _ = load_synthetic()
        _model = LinearRegression().fit(Xs.values, ys.values)
    return _model


def parse_request(data):
    thrust_gf = float(data["thrust"])
    diameter = float(data["diameter"])
    rpm = float(data["rpm"])
    blades = data.get("blades") or None
    blades = int(blades) if blades else None
    if not (0 < thrust_gf <= 50000):
        raise ValueError("thrust must be between 0 and 50000 gf")
    if not (1 <= diameter <= 40):
        raise ValueError("diameter must be between 1 and 40 inches")
    if not (100 <= rpm <= 60000):
        raise ValueError("rpm must be between 100 and 60000")
    return thrust_gf, diameter, rpm, blades


def warnings_for(diameter, result):
    out = []
    if not 2 <= diameter <= 30:
        out.append(f"{diameter:g} in is outside SCOPE.md's 2-30 inch design class.")
    elif diameter < 6 or diameter > 11:
        out.append(
            "Phase 3 validated efficiency to within ~15% of the best real propeller "
            "only for 6-11 inch diameters. Outside that the model extrapolates below "
            "its training Reynolds range and did markedly worse.")
    if not result["meets_thrust"]:
        err = result["thrust_error_pct"]
        if err < 0:
            out.append(
                f"No blade count reached the requested thrust -- the closest falls "
                f"{abs(err):.0f}% short. Since the target rpm is a motor ceiling rather "
                f"than a knob, this design would not meet the requirement. Try a "
                f"larger diameter or higher rpm.")
        else:
            out.append(
                f"The best available candidate overshoots by {err:.0f}%. It meets the "
                f"thrust requirement but is oversized -- heavier and drawing more "
                f"current than the mission needs. A smaller diameter or lower rpm "
                f"would fit the requirement more closely.")
    return out


@app.route("/")
def index():
    return render_template("index.html", candidates=list(CANDIDATE_BLADES))


@app.route("/api/design", methods=["POST"])
def api_design():
    try:
        thrust_gf, diameter, rpm, blades = parse_request(request.json)
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    r = design(get_model(), thrust_gf * 9.80665 / 1000.0, diameter, rpm, blades)
    R_in = diameter / 2.0
    return jsonify({
        "blade_count": r["blade_count"],
        "thrust_gf": r["thrust_N"] * GF_PER_N,
        "thrust_error_pct": r["thrust_error_pct"],
        "shaft_power_W": r["shaft_power_W"],
        "gf_per_W": r["thrust_per_watt_gf_W"],
        "CT": r["CT"], "CP": r["CP"],
        "meets_thrust": r["meets_thrust"],
        "warnings": warnings_for(diameter, r),
        "candidates": [
            {"blades": c["blade_count"], "thrust_gf": c["thrust_N"] * GF_PER_N,
             "error_pct": c["thrust_error_pct"], "watts": c["shaft_power_W"],
             "gf_per_W": c["thrust_per_watt_gf_W"], "meets": c["meets_thrust"]}
            for c in r.get("candidates", [])],
        "geometry": [
            {"r_R": x, "c_R": c, "chord_in": c * R_in, "twist_deg": b}
            for x, c, b in zip(STATIONS, r["c_R"], r["beta_deg"])],
    })


@app.route("/api/cad", methods=["POST"])
def api_cad():
    try:
        thrust_gf, diameter, rpm, blades = parse_request(request.json)
        bore = float(request.json.get("bore", 6.0))
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    r = design(get_model(), thrust_gf * 9.80665 / 1000.0, diameter, rpm, blades)
    prop = build_propeller(r["c_R"], r["beta_deg"], diameter, r["blade_count"],
                           bore_mm=bore)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"prop_{diameter:g}in_{thrust_gf:g}gf_{r['blade_count']}b_{uuid.uuid4().hex[:6]}"
    paths = export(prop, stem, out_dir=OUT)
    stl = next(p for p in paths if p.suffix == ".stl")
    png = render(stl)
    mesh = check_manifold(stl)
    solid = prop.val()

    return jsonify({
        "stem": stem,
        "volume_cm3": solid.Volume() / 1000.0,
        "mass_pla_g": solid.Volume() / 1000.0 * 1.24,
        "watertight": mesh["watertight"],
        "triangles": mesh["triangles"],
        "png": f"/files/{png.name}",
        "stl": f"/files/{stl.name}",
        "step": f"/files/{stem}.step",
    })


@app.route("/files/<path:name>")
def files(name):
    return send_from_directory(OUT, name, as_attachment=name.endswith((".stl", ".step")))


if __name__ == "__main__":
    print("fitting model...")
    get_model()
    print("ready -> http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
