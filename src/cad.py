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


def bore_profile(bore_mm, flat_mm=None, n=72):
    """Bore cross-section: a circle, optionally with a D-flat.

    A plain round bore transmits torque only by friction, and in PLA that is
    not a mounting scheme. At the 10 inch design point the shaft carries about
    81 mN*m, which a 6 mm round bore would need roughly 90 N of radial grip to
    hold -- far more than a printed press fit provides, and PLA creeps under
    sustained load so whatever grip exists fades within hours.

    A D-flat matching the motor shaft transmits torque through GEOMETRY
    instead. It cannot creep loose, and it fails safe: if it ever does slip it
    slips one flat and re-engages rather than wallowing the bore round.

    Args:
        flat_mm: distance from the shaft axis to the flat. Measure the motor
            shaft across the flat and halve it. None gives a plain round bore.

    Returns:
        list of (x, y) points, or None for a plain circle.
    """
    r = bore_mm / 2.0
    if flat_mm is None or flat_mm >= r or flat_mm <= 0:
        return None
    th0 = math.acos(min(1.0, flat_mm / r))
    angs = np.linspace(th0, 2 * math.pi - th0, n)
    return [(float(r * math.cos(a)), float(r * math.sin(a))) for a in angs]


def mount_report(bore_mm, seat_dia_mm, clamp_len_mm, hub_dia_mm):
    """Flag mounting geometry that will not survive assembly or flight."""
    issues = []
    seat_width = (seat_dia_mm - bore_mm) / 2.0
    if seat_width < 2.0:
        issues.append(f"washer seat only {seat_width:.1f} mm wide -- the nut will "
                      f"dig into it; widen seat_dia_mm")
    if clamp_len_mm < 6.0:
        issues.append(f"clamp collar only {clamp_len_mm:.1f} mm long -- too short for "
                      f"most prop adapters to grip; raise clamp_len_mm")
    wall = (hub_dia_mm - seat_dia_mm) / 2.0
    if wall < 3.0:
        issues.append(f"only {wall:.1f} mm of hub wall outside the seat -- the hub may "
                      f"split; reduce seat_dia_mm or widen the hub")
    return issues


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
                    bore_mm=6.0, hub_dia_mm=None, hub_thick_mm=None,
                    shaft_flat_mm=None, clamp_len_mm=10.0, seat_dia_mm=None,
                    report=None):
    """Full propeller: blades patterned around a hub that can actually be mounted.

    Args:
        bore_mm: motor shaft diameter. 5, 6 and 8 mm cover most brushless
            outrunners in this class; 6 is the common default.
        hub_dia_mm: hub outer diameter. Defaults to just past the first
            station so the blade root is embedded in solid material rather
            than meeting it at a seam.
        hub_thick_mm: hub height along the shaft.
        shaft_flat_mm: distance from shaft axis to the flat, for a D-bore.
            None gives a round bore, which relies on friction alone -- see
            bore_profile for why that does not hold in PLA.
        clamp_len_mm: length of the collar the prop adapter actually grips.
            The hub has to be tall to contain a twisted root section -- 27 mm
            on the 10 inch design -- but standard adapters clamp only 10-15 mm,
            so a bare hub that thick physically will not fit one. Both faces
            are counterbored to leave a collar of this length.
        seat_dia_mm: diameter of those counterbores. Their floors are the flat,
            parallel annular faces the washer and adapter shoulder bear on.
            Face clamping at ~8 mm radius needs only about 34 N to carry the
            81 mN*m this design produces, which a hand-tight nut supplies
            several times over -- but only if the seat is genuinely flat, which
            a bare cylinder does not guarantee once blade roots merge into it.
        report: optional list; mounting warnings are appended to it.

    MOUNTING DIRECTION: fit the propeller so shaft torque TIGHTENS the nut.
    Mounted the other way it unwinds itself in flight.
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

    # Counterbore both faces to leave a short clamp collar, before the bore is
    # cut. Capped at 75% of the hub so the recess cannot eat into blade roots.
    if seat_dia_mm is None:
        seat_dia_mm = max(2.2 * bore_mm, 14.0)
    seat_dia_mm = min(seat_dia_mm, 0.75 * hub_dia_mm)
    collar = min(clamp_len_mm, hub_thick_mm)

    if collar < hub_thick_mm:
        d = (hub_thick_mm - collar) / 2.0
        for z, h in ((hub_thick_mm / 2.0 - d, d + 1.0),
                     (-hub_thick_mm / 2.0 - 1.0, d + 1.0)):
            prop = prop.cut(cq.Workplane("XY").workplane(offset=z)
                            .circle(seat_dia_mm / 2.0).extrude(h))

    if report is not None:
        report.extend(mount_report(bore_mm, seat_dia_mm, collar, hub_dia_mm))

    # Bore last, so no blade union or counterbore can fill it back in.
    pts = bore_profile(bore_mm, shaft_flat_mm)
    cutter = (cq.Workplane("XY").polyline(pts).close() if pts
              else cq.Workplane("XY").circle(bore_mm / 2.0))
    prop = prop.cut(cutter.extrude(hub_thick_mm, both=True))
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


def render(stl_path, png_path=None):
    """Render an exported STL to a PNG: top, side and iso views.

    Useful because the two things most likely to be wrong are visible at a
    glance and invisible in the numbers -- twist running the wrong way, and a
    blade that failed to fuse with the hub.
    """
    import struct

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    stl_path = Path(stl_path)
    png_path = Path(png_path or stl_path.with_suffix(".png"))

    data = stl_path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    tris = np.zeros((n, 3, 3))
    off = 84
    for i in range(n):
        v = struct.unpack("<12fH", data[off:off + 50])
        tris[i] = np.array(v[3:12]).reshape(3, 3)
        off += 50

    lim = float(np.abs(tris).max()) * 1.05
    fig = plt.figure(figsize=(15, 5))
    for k, (elev, azim, title) in enumerate([
            (90, -90, "top (rotor disc)"),
            (0, -90, "side (twist visible)"),
            (35, -60, "iso")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            tris, facecolor="#7aa6d6", edgecolor="#2b4a6f", linewidth=0.05))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=110, facecolor="white")
    plt.close(fig)
    return png_path


def check_manifold(stl_path):
    """Is the mesh watertight? Every edge must be shared by exactly two faces.

    A slicer will either refuse a non-manifold mesh or silently produce a part
    with holes in it, so this is worth checking before printing rather than
    after.
    """
    import struct
    from collections import Counter

    data = Path(stl_path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    edges = Counter()
    off = 84
    for _ in range(n):
        v = struct.unpack("<12fH", data[off:off + 50])
        pts = [tuple(round(c, 4) for c in v[3:6]),
               tuple(round(c, 4) for c in v[6:9]),
               tuple(round(c, 4) for c in v[9:12])]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[frozenset((pts[a], pts[b]))] += 1
        off += 50
    bad = sum(1 for c in edges.values() if c != 2)
    return {"triangles": n, "edges": len(edges), "bad_edges": bad,
            "watertight": bad == 0}


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
    ap.add_argument("--shaft-flat", type=float, default=None,
                    help="distance from shaft axis to its flat, mm -- gives a D-bore "
                         "that drives on geometry instead of friction (measure the "
                         "shaft across the flat and halve it)")
    ap.add_argument("--clamp-len", type=float, default=10.0,
                    help="length of collar the prop adapter grips, mm (default 10)")
    ap.add_argument("--seat-dia", type=float, default=None,
                    help="counterbore diameter for the washer seats, mm")
    ap.add_argument("--hub-dia", type=float, default=None, help="hub outer diameter, mm")
    ap.add_argument("--hub-thick", type=float, default=None, help="hub height, mm")
    ap.add_argument("--name", default=None, help="output filename stem")
    ap.add_argument("--render", action="store_true",
                    help="also write a PNG with top, side and iso views")
    ap.add_argument("--check", action="store_true",
                    help="verify the STL mesh is watertight before printing")
    a = ap.parse_args()

    thrust_N = a.thrust * 9.80665 / 1000.0
    Xs, ys, _ = load_synthetic()
    model = LinearRegression().fit(Xs.values, ys.values)
    r = design(model, thrust_N, a.diameter, a.rpm, a.blades)

    print(f"design  : {r['blade_count']} blades, "
          f"{r['thrust_N'] * GF_PER_N:.0f} gf ({r['thrust_error_pct']:+.1f}%), "
          f"{r['shaft_power_W']:.1f} W, {r['thrust_per_watt_gf_W']:.2f} gf/W")

    issues = []
    prop = build_propeller(r["c_R"], r["beta_deg"], a.diameter, r["blade_count"],
                           bore_mm=a.bore, hub_dia_mm=a.hub_dia,
                           hub_thick_mm=a.hub_thick, shaft_flat_mm=a.shaft_flat,
                           clamp_len_mm=a.clamp_len, seat_dia_mm=a.seat_dia,
                           report=issues)
    solid = prop.val()

    drive = f"D-flat at {a.shaft_flat} mm" if a.shaft_flat else "ROUND (friction only)"
    print(f"mount   : {a.bore} mm bore, {drive}, {a.clamp_len:g} mm clamp collar")
    if not a.shaft_flat:
        print("          note: a round bore holds by friction alone. PLA creeps, so")
        print("          it will loosen -- pass --shaft-flat to drive on geometry.")
    for msg in issues:
        print(f"          WARNING: {msg}")
    bb = solid.BoundingBox()
    print(f"solid   : valid={solid.isValid()}  volume={solid.Volume() / 1000:.1f} cm^3")
    print(f"          {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm, bore {a.bore} mm")

    # PLA at ~1.24 g/cm^3, roughly what it will weigh printed solid
    print(f"          ~{solid.Volume() / 1000 * 1.24:.0f} g in PLA if printed solid")

    stem = a.name or f"prop_{a.diameter:g}in_{a.thrust:g}gf_{r['blade_count']}b"
    paths = export(prop, stem)
    for p in paths:
        print(f"wrote   : {p}  ({p.stat().st_size / 1000:.0f} kB)")

    stl = next(p for p in paths if p.suffix == ".stl")
    if a.check:
        c = check_manifold(stl)
        print(f"mesh    : {c['triangles']} triangles, {c['edges']} edges, "
              f"watertight={c['watertight']}"
              + (f"  ({c['bad_edges']} bad edges!)" if c["bad_edges"] else ""))
    if a.render:
        print(f"wrote   : {render(stl)}")


if __name__ == "__main__":
    main()
