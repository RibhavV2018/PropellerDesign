"""Turn predicted blade geometry into a printable 3D solid.

Phase 4. Takes the chord and twist distribution the model predicts and lofts
it into a propeller with a motor-mount hub, exported as STEP and STL.

This is the step the original AIAA paper flagged as future work and never did,
and it is what turns a data-science script into a manufacturable part.

COORDINATE SYSTEM

    Z   rotor axis (thrust direction)
    X   radial, along the blade span
    Y   tangential, the direction of rotation

Each station's airfoil sits in the plane perpendicular to X, with its chord
line rotated out of the Y axis by the local twist angle. Sections are stacked
on the QUARTER CHORD rather than the leading edge, which is how real blades
are built -- it keeps the aerodynamic centre on a straight axis instead of
sweeping it backwards as chord changes.

THE SECTION

An analytic NACA 4-digit profile rather than a tabulated Clark-Y. Thickness
becomes a parameter that way, which matters: real propellers are thick at the
root for bending strength and thin at the tip for efficiency, and a fixed
coordinate table cannot express that.

A HONEST GAP: the BEMT solver used a generic low-Reynolds airfoil model, not
this specific profile. The aerodynamics and the geometry therefore do not use
identical sections. That is unavoidable -- the UIUC database never names the
real blade sections either, which is the same limitation that sets the 17%
validation floor -- but it should be stated rather than glossed over.

THE ROOT SINGULARITY

Constant-pitch twist is arctan(P / 2*pi*r), which tends to 90 degrees as r
tends to 0. Stations start at r/R = 0.15 where twist is steep but finite, and
the hub occupies everything inboard of that, so the singularity is never
evaluated. Do not extend the stations inward without adding a blend.
"""

import math
import sys
from pathlib import Path

import cadquery as cq
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import STATIONS

IN_MM = 25.4

# Thickness-to-chord at root and tip. Real blades taper: strength inboard
# where bending moment is highest, thinness outboard where it costs drag.
THICKNESS_ROOT = 0.14
THICKNESS_TIP = 0.07

CAMBER = 0.04          # NACA 4-digit max camber, 4% -- typical for a prop
CAMBER_POS = 0.4       # its chordwise position
N_POINTS = 60          # points per airfoil section


def naca4(thickness, camber=CAMBER, camber_pos=CAMBER_POS, n=N_POINTS):
    """NACA 4-digit section as a closed loop of (chordwise, normal) points.

    Analytic, so thickness and camber can vary station to station. Cosine
    spacing puts more points near the leading edge, where curvature is
    highest and a uniform spacing would visibly facet the print.

    Returns points ordered around the section: upper surface trailing edge to
    leading edge, then lower surface back to trailing edge.
    """
    beta = np.linspace(0.0, np.pi, n)
    x = (1 - np.cos(beta)) / 2.0                      # cosine spacing, 0..1

    m, p, t = camber, camber_pos, thickness
    yc = np.where(x < p,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p,
                   2 * m / p**2 * (p - x),
                   2 * m / (1 - p)**2 * (p - x))
    theta = np.arctan(dyc)

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)

    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    # upper surface TE->LE, then lower LE->TE, skipping the duplicated LE point
    px = np.concatenate([xu[::-1], xl[1:]])
    py = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([px, py])


def station_wire(r_mm, chord_mm, twist_deg, thickness):
    """One airfoil section, placed and twisted at its radius.

    The section is built in chord/normal coordinates, shifted so the quarter
    chord sits at the origin, rotated by the twist angle, and then mapped into
    the Y-Z plane at X = r.
    """
    prof = naca4(thickness)
    u = (prof[:, 0] - 0.25) * chord_mm      # chordwise, quarter-chord datum
    v = prof[:, 1] * chord_mm               # normal to chord

    a = math.radians(twist_deg)
    y = u * math.cos(a) - v * math.sin(a)   # tangential
    z = u * math.sin(a) + v * math.cos(a)   # axial

    pts = [cq.Vector(float(r_mm), float(yy), float(zz)) for yy, zz in zip(y, z)]
    return cq.Wire.makePolygon(pts, close=True)


def build_blade(c_R, beta_deg, diameter_in):
    """Loft the station sections into a single blade solid."""
    R_mm = diameter_in * IN_MM / 2.0
    wires = []
    for x, c, b in zip(STATIONS, c_R, beta_deg):
        # thickness tapers linearly root to tip, in t/c terms
        f = (x - STATIONS[0]) / (STATIONS[-1] - STATIONS[0])
        t = THICKNESS_ROOT + (THICKNESS_TIP - THICKNESS_ROOT) * f
        wires.append(station_wire(x * R_mm, c * R_mm, b, t))
    return cq.Solid.makeLoft(wires)


def build_propeller(c_R, beta_deg, diameter_in, blade_count,
                    bore_mm=6.0, hub_dia_mm=None, hub_thick_mm=None):
    """Full propeller: blades patterned around a bored hub.

    Args:
        bore_mm: motor shaft diameter. 5, 6 and 8 mm cover most brushless
            outrunners in this class; 6 is the common default.
        hub_dia_mm: hub outer diameter. Defaults to just past the first
            station so the blade root is embedded in solid material rather
            than meeting it at a seam.
        hub_thick_mm: hub height along the shaft.
    """
    R_mm = diameter_in * IN_MM / 2.0
    root_r = STATIONS[0] * R_mm

    # Hub diameter: only a small overlap past the first station. A generous
    # hub looks safer and is not -- every millimetre of extra radius buries
    # blade span that BEMT modelled as lifting surface, so the built part
    # makes less thrust than the design predicted. 1.15x keeps the union
    # solid without eating much span.
    if hub_dia_mm is None:
        hub_dia_mm = 2.0 * root_r * 1.15

    # Hub thickness: enough to contain the ROOT SECTION, computed from its
    # actual axial extent rather than guessed from chord. A twisted section
    # projects chord*sin(twist) along the shaft, plus its own thickness.
    if hub_thick_mm is None:
        c0, b0 = c_R[0] * R_mm, math.radians(beta_deg[0])
        axial = abs(c0 * math.sin(b0)) + THICKNESS_ROOT * c0 * abs(math.cos(b0))
        hub_thick_mm = max(6.0, axial * 1.1)

    blade = build_blade(c_R, beta_deg, diameter_in)

    hub = (cq.Workplane("XY")
           .circle(hub_dia_mm / 2.0)
           .extrude(hub_thick_mm / 2.0, both=True))

    prop = hub
    for i in range(int(blade_count)):
        rotated = blade.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1),
                               i * 360.0 / blade_count)
        prop = prop.union(cq.Workplane(obj=rotated))

    # bore last, so no blade union can fill it back in
    prop = prop.cut(cq.Workplane("XY").circle(bore_mm / 2.0)
                    .extrude(hub_thick_mm, both=True))
    return prop


def export(prop, stem, out_dir=None):
    """Write STEP (CAD interchange) and STL (printing)."""
    out_dir = Path(out_dir or Path(__file__).resolve().parent.parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("step", "stl"):
        p = out_dir / f"{stem}.{ext}"
        cq.exporters.export(prop, str(p))
        paths.append(p)
    return paths


# --- command line -----------------------------------------------------------

def main() -> None:
    import argparse

    from sklearn.linear_model import LinearRegression

    from dataset import load_synthetic
    from design import design
    from physics import GF_PER_N

    ap = argparse.ArgumentParser(
        description="Design a propeller and export it as STEP and STL.",
        epilog="example: python src/cad.py --thrust 500 --diameter 10 --rpm 6000",
    )
    ap.add_argument("--thrust", type=float, required=True, help="target thrust, grams-force")
    ap.add_argument("--diameter", type=float, required=True, help="diameter, inches")
    ap.add_argument("--rpm", type=float, required=True, help="target shaft speed")
    ap.add_argument("--blades", type=int, default=None, help="fix blade count (default: auto 2-6)")
    ap.add_argument("--bore", type=float, default=6.0, help="motor shaft diameter, mm (default 6)")
    ap.add_argument("--hub-dia", type=float, default=None, help="hub outer diameter, mm")
    ap.add_argument("--hub-thick", type=float, default=None, help="hub height, mm")
    ap.add_argument("--name", default=None, help="output filename stem")
    a = ap.parse_args()

    thrust_N = a.thrust * 9.80665 / 1000.0
    Xs, ys, _ = load_synthetic()
    model = LinearRegression().fit(Xs.values, ys.values)
    r = design(model, thrust_N, a.diameter, a.rpm, a.blades)

    print(f"design  : {r['blade_count']} blades, "
          f"{r['thrust_N'] * GF_PER_N:.0f} gf ({r['thrust_error_pct']:+.1f}%), "
          f"{r['shaft_power_W']:.1f} W, {r['thrust_per_watt_gf_W']:.2f} gf/W")

    prop = build_propeller(r["c_R"], r["beta_deg"], a.diameter, r["blade_count"],
                           bore_mm=a.bore, hub_dia_mm=a.hub_dia,
                           hub_thick_mm=a.hub_thick)
    solid = prop.val()
    bb = solid.BoundingBox()
    print(f"solid   : valid={solid.isValid()}  volume={solid.Volume() / 1000:.1f} cm^3")
    print(f"          {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm, bore {a.bore} mm")

    # PLA at ~1.24 g/cm^3, roughly what it will weigh printed solid
    print(f"          ~{solid.Volume() / 1000 * 1.24:.0f} g in PLA if printed solid")

    stem = a.name or f"prop_{a.diameter:g}in_{a.thrust:g}gf_{r['blade_count']}b"
    for p in export(prop, stem):
        print(f"wrote   : {p}  ({p.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
