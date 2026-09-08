"""Blade element momentum theory solver for hover (J = 0).

Computes the thrust and torque a given blade geometry produces, which is the
FORWARD direction: geometry -> performance. The ML model runs the inverse.
Two uses: generating synthetic training data (Phase 1b), and grading
ML-predicted blades that nobody has built (Phase 3).

THE IDEA
Thrust on one annulus is computed two independent ways, and the physical
answer is where they agree.

  Momentum:       dT = 4*pi*rho*r*F*v_i^2 * dr
                  Depends only on induced velocity. Blade shape is invisible.

  Blade element:  dT = B * (dL*cos(phi) - dD*sin(phi))
                  with dL = 0.5*rho*U^2*c*Cl*dr
                  Depends on chord, twist, blade count, airfoil -- and on
                  v_i too, through the inflow angle.

The circularity is physical, not a modelling artifact: more thrust pulls more
air through, which raises inflow, which lowers angle of attack, which cuts
thrust. A real propeller sits at that equilibrium.

GEOMETRY AT ONE STATION (hover, so no freestream axial term)

    U_T = Omega * r              tangential, from rotation
    U_P = v_i                    axial, the induced inflow
    phi = atan2(U_P, U_T)        inflow angle
    alpha = theta - phi          angle of attack: twist MINUS inflow
    U   = sqrt(U_T^2 + U_P^2)    what the section actually sees

Textbook BEMT carries a freestream velocity and a climb inflow ratio through
the algebra. Both are zero here. If a derivation you are reading has terms
this module lacks, that is why -- do not add them back.

    ./.venv/bin/python src/bemt.py
"""

import math

RHO = 1.225        # kg/m^3, ISA sea level -- matches physics.py and SCOPE.md
MU = 1.81e-5       # Pa*s, dynamic viscosity of air


def placeholder_airfoil(alpha_rad, re):
    """A deliberately crude airfoil model, so `residual` can be tested alone.

    Thin-airfoil lift slope with a hard stall cutoff, and constant drag. It
    ignores Reynolds number entirely, which is wrong -- your propellers run at
    a median Re of 43,000, where drag rises sharply as Re falls. Replacing
    this is the next piece of work.

    Returns:
        (cl, cd)
    """
    STALL = math.radians(12.0)
    a = max(-STALL, min(STALL, alpha_rad))
    return 2.0 * math.pi * a, 0.02


def tip_loss_factor(r, R, phi, B):
    """Prandtl tip-loss factor F, in [0, 1].

        F = (2/pi) * arccos(exp(-f)),   f = (B/2) * (R - r) / (r * sin(phi))

    Air escapes around the blade tip rather than being pushed down, so the
    outboard sections carry less load than 2D theory predicts. F -> 1 inboard
    and F -> 0 at the tip. Without it BEMT over-predicts hover thrust by
    roughly 5-15%.

    F depends on phi, which depends on v_i, so this must be recomputed inside
    every residual evaluation -- not hoisted out of the loop.
    """
    s = math.sin(abs(phi))
    if s < 1e-9 or r <= 0:
        return 1.0
    f = (B / 2.0) * (R - r) / (r * s)
    if f > 30:            # exp(-f) underflows; F is 1 to machine precision
        return 1.0
    return (2.0 / math.pi) * math.acos(min(1.0, math.exp(-f)))


def residual(v_i, r, R, chord, theta_rad, omega, B, airfoil=placeholder_airfoil):
    """Disagreement between blade-element and momentum thrust, per unit span.

    The physical induced velocity is the v_i where this returns zero.

    Args:
        v_i: induced (axial) velocity at this station, m/s -- THE UNKNOWN.
        r: radius of this station, m.
        R: tip radius, m.
        chord: chord at this station, m.
        theta_rad: blade angle (twist) at this station, RADIANS.
                   Your `beta_*` columns are in degrees -- convert first.
        omega: shaft speed, RADIANS PER SECOND (not rpm, not rev/s).
        B: blade count.
        airfoil: callable (alpha_rad, re) -> (cl, cd).

    Returns:
        blade_element_thrust_per_span - momentum_thrust_per_span, in N/m.
        The `dr` cancels because both sides are per unit span.

    TODO(ribhav): implement. In order:

      1. U_T = omega * r,  U_P = v_i
      2. phi = atan2(U_P, U_T),  alpha = theta_rad - phi
      3. U = hypot(U_T, U_P)
      4. Re = RHO * U * chord / MU        <- local, varies along the blade
      5. cl, cd = airfoil(alpha, Re)
      6. Blade element side, per unit span:
             B * 0.5*RHO*U**2*chord * (cl*cos(phi) - cd*sin(phi))
      7. Momentum side, per unit span, with F = tip_loss_factor(r, R, phi, B):
             4*math.pi*RHO*r*F*v_i**2
      8. Return (6) - (7)

    Note the sign convention in step 6: lift tilts BACK by phi, so it
    contributes cl*cos(phi), while drag subtracts. Getting that backwards
    gives a solver that converges happily to the wrong answer.
    """

    U_T = omega * r
    U_P = v_i
    phi = math.atan2(U_P, U_T)
    alpha = theta_rad - phi
    U = math.hypot(U_T, U_P)
    Re = RHO * U * chord / MU
    cl, cd = airfoil(alpha, Re)
    F = tip_loss_factor(r, R, phi, B)
    return (B*0.5*RHO*(U**2)*chord * (cl*math.cos(phi) - cd*math.sin(phi))) - (4*math.pi*RHO*r*F*(v_i**2))


def bisect(fn, lo, hi, tol=1e-10, max_iter=200):
    """Find x in [lo, hi] where fn(x) == 0, by repeated halving.

    Bisection rather than fixed-point iteration on v_i. Fixed-point is the
    textbook presentation and is faster when it works, but it oscillates or
    diverges near stall and at low inflow -- precisely where these low-Re
    micro-propellers operate. Bisection cannot fail once the root is
    bracketed, needs no relaxation factor, and costs nothing at this scale.

    The bracket is guaranteed by physics, not luck:
        v_i -> 0     : alpha -> theta, blade element thrust is positive while
                       momentum thrust vanishes           -> residual > 0
        v_i large    : alpha goes negative, lift reverses, momentum term grows
                                                          -> residual < 0
    """
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"root not bracketed: f({lo:.4g})={f_lo:.4g}, f({hi:.4g})={f_hi:.4g}"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)



def solve_station(r, R, chord, theta_rad, omega, B, airfoil=placeholder_airfoil):
    """Converge one radial station and return its thrust and torque per span.

    Args:
        (same as `residual`)

    Returns:
        dict with:
            v_i      -- converged induced velocity, m/s
            dT_dr    -- thrust per unit span at this station, N/m
            dQ_dr    -- torque per unit span at this station, N*m/m
            phi, alpha, cl, cd, Re   -- diagnostics, for when validation
                                        disagrees and you need to see why

    The tip station is special-cased. At r = R the Prandtl factor is exactly
    zero for any inflow angle, so the momentum sink vanishes, the residual
    never changes sign, and bisection has no root to find. Physically that IS
    the answer -- the tip carries no load, because pressure equalizes around
    it -- so the station returns zero thrust and torque rather than being
    nudged past the singularity with a clamped F. Chord goes to nearly zero
    there as well, so nothing real is being discarded.
    """
    if R - r < 1e-9:
        return {"v_i": 0.0, "dT_dr": 0.0, "dQ_dr": 0.0,
                "phi": 0.0, "alpha": 0.0, "cl": 0.0, "cd": 0.0, "Re": 0.0}

    def f(v):
        return residual(v, r, R, chord, theta_rad, omega, B, airfoil)

    lo, hi = 1e-6, omega * r
    try:
        v_i = bisect(f, lo, hi)
    except ValueError as exc:                     # not silently swallowed
        raise ValueError(f"station r={r:.4f} m (r/R={r/R:.3f}): {exc}") from exc

    U_T, U_P = omega * r, v_i
    phi = math.atan2(U_P, U_T)
    alpha = theta_rad - phi
    U = math.hypot(U_T, U_P)
    Re = RHO * U * chord / MU
    cl, cd = airfoil(alpha, Re)

    q = 0.5 * RHO * U**2 * chord                  # dynamic pressure * chord
    # Same lift and drag, resolved onto perpendicular axes. Drag SUBTRACTS
    # from thrust and ADDS to torque -- that asymmetry is the cost of spinning
    # a propeller. Torque carries an extra r because it is a moment.
    dT_dr = B * q * (cl * math.cos(phi) - cd * math.sin(phi))
    dQ_dr = B * r * q * (cl * math.sin(phi) + cd * math.cos(phi))

    return {"v_i": v_i, "dT_dr": dT_dr, "dQ_dr": dQ_dr,
            "phi": phi, "alpha": alpha, "cl": cl, "cd": cd, "Re": Re}


def solve_propeller(r_R, c_R, beta_deg, diameter_in, rpm, blade_count,
                    airfoil=placeholder_airfoil, stations=False):
    """Run every station and integrate to whole-propeller performance.

    Args:
        r_R: radial stations as fractions of tip radius, ascending.
        c_R: chord at each station, normalized by tip radius R.
        beta_deg: blade angle at each station, DEGREES.
        diameter_in: propeller diameter, INCHES.
        rpm: shaft speed.
        blade_count: number of blades.
        airfoil: callable (alpha_rad, re) -> (cl, cd).
        stations: if True, include the per-station solutions in the result.

    Returns:
        dict with thrust_N, torque_Nm, shaft_power_W, CT, CP.

    Stations begin at r/R = 0.15, so the hub region contributes nothing. Chord
    is small there and it sits inside the spinner, but it is an assumption
    worth stating rather than leaving implicit.
    """
    R = diameter_in * IN_M / 2.0
    omega = 2.0 * math.pi * rpm / 60.0

    rs, sols = [], []
    for x, c, b in zip(r_R, c_R, beta_deg):
        r = x * R
        rs.append(r)
        sols.append(solve_station(r, R, c * R, math.radians(b), omega,
                                  blade_count, airfoil))

    # Trapezoid rule over the stations. Written out rather than reaching for
    # numpy so the integration is visible and version-independent.
    T = Q = 0.0
    for i in range(len(rs) - 1):
        dr = rs[i + 1] - rs[i]
        T += 0.5 * (sols[i]["dT_dr"] + sols[i + 1]["dT_dr"]) * dr
        Q += 0.5 * (sols[i]["dQ_dr"] + sols[i + 1]["dQ_dr"]) * dr

    P = Q * omega
    n, D = rpm / 60.0, diameter_in * IN_M

    # Inverts the formulas in physics.py, so solver output and measured data
    # land in the same units and compare directly.
    out = {
        "thrust_N": T,
        "torque_Nm": Q,
        "shaft_power_W": P,
        "CT": T / (RHO * n**2 * D**4),
        "CP": P / (RHO * n**3 * D**5),
    }
    if stations:
        out["stations"] = sols
    return out


# --- checks -----------------------------------------------------------------
# One real station: APC 9x4.7 SF at r/R = 0.75, 6026 rpm -- the same propeller
# and operating point physics.py is verified against.

IN_M = 0.0254
_R = 9.0 * IN_M / 2          # tip radius, m
_r = 0.75 * _R               # the 0.75R reference station
_chord = 0.199 * _R          # c_R_r075 from the dataset
_theta = math.radians(11.56)  # beta_r075, degrees -> radians
_omega = 2 * math.pi * 6026 / 60
_B = 2


def _res(v):
    return residual(v, _r, _R, _chord, _theta, _omega, _B)


def check_residual() -> int:
    checks = []

    # The bracket argument, verified rather than assumed.
    lo, hi = 1e-6, _omega * _r
    checks.append(("residual > 0 as v_i -> 0", _res(lo) > 0, f"{_res(lo):+.3f} N/m"))
    checks.append(("residual < 0 at v_i = omega*r", _res(hi) < 0, f"{_res(hi):+.3f} N/m"))

    # Monotone decreasing: more inflow always means less net thrust. If this
    # fails there is a sign error somewhere in the velocity triangle.
    vs = [lo + (hi - lo) * k / 40 for k in range(41)]
    rs = [_res(v) for v in vs]
    checks.append(("residual decreases monotonically",
                   all(a > b for a, b in zip(rs, rs[1:])), "40 samples"))

    v = bisect(_res, lo, hi)
    phi = math.atan2(v, _omega * _r)
    alpha = _theta - phi
    checks.append(("bisection converges", abs(_res(v)) < 1e-6, f"|R| = {abs(_res(v)):.2e}"))

    # Momentum theory over the whole disk says the mean induced velocity for
    # this propeller's measured 4.02 N is sqrt(T / (2*rho*A)) = 6.3 m/s. The
    # local value at 0.75R should be the same order, not wildly off.
    checks.append(("v_i is physically plausible (2-15 m/s)", 2 < v < 15, f"{v:.2f} m/s"))

    # A real propeller section runs at a few degrees of attack, well below
    # stall. Double digits means the velocity triangle is wrong.
    checks.append(("alpha is a sane few degrees (0-10)",
                   0 < math.degrees(alpha) < 10, f"{math.degrees(alpha):.2f} deg"))

    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<38} {detail}")
    print(f"  -> converged: v_i = {v:.3f} m/s, phi = {math.degrees(phi):.2f} deg, "
          f"alpha = {math.degrees(alpha):.2f} deg")
    return sum(1 for _, ok, _ in checks if not ok)


def check_propeller() -> int:
    """Run the whole APC 9x4.7 SF blade and compare against measurement."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd

    df = pd.read_csv(Path(__file__).resolve().parent.parent
                     / "data" / "processed" / "uiuc_hover.csv")
    row = df[(df.prop_name == "apcsf_9x4.7") & (df.rpm == 6026.0)].iloc[0]

    stations = [round(x, 2) for x in [0.15 + 0.05 * k for k in range(18)]]
    r_R = stations
    c_R = [row[f"c_R_r{int(round(x*100)):03d}"] for x in stations]
    beta = [row[f"beta_r{int(round(x*100)):03d}"] for x in stations]

    out = solve_propeller(r_R, c_R, beta, row.diameter_in, row.rpm, int(row.blade_count))

    checks = []
    checks.append(("CT positive", out["CT"] > 0, f"{out['CT']:.4f}"))
    checks.append(("CP positive", out["CP"] > 0, f"{out['CP']:.4f}"))

    # Internal consistency: CT must reproduce the thrust it was derived from.
    n, D = row.rpm / 60.0, row.diameter_in * IN_M
    t_from_ct = out["CT"] * RHO * n**2 * D**4
    checks.append(("CT reproduces thrust", abs(t_from_ct - out["thrust_N"]) < 1e-9,
                   f"{t_from_ct:.4f} vs {out['thrust_N']:.4f} N"))

    # CT is a coefficient: it should be nearly RPM-independent. Any strong
    # trend means rpm is leaking into the answer where it should cancel.
    out_lo = solve_propeller(r_R, c_R, beta, row.diameter_in, row.rpm * 0.6, int(row.blade_count))
    drift = abs(out_lo["CT"] - out["CT"]) / out["CT"]
    checks.append(("CT ~ rpm-independent (<15% over 0.6x)", drift < 0.15, f"{drift*100:.1f}%"))

    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<38} {detail}")

    print(f"\n  MEASURED : CT = {row.CT:.4f}  CP = {row.CP:.4f}  "
          f"T = {row.thrust_N:.3f} N  P = {row.shaft_power_W:.2f} W")
    print(f"  SOLVER   : CT = {out['CT']:.4f}  CP = {out['CP']:.4f}  "
          f"T = {out['thrust_N']:.3f} N  P = {out['shaft_power_W']:.2f} W")
    print(f"  error    : CT {(out['CT']/row.CT-1)*100:+.1f}%   CP {(out['CP']/row.CP-1)*100:+.1f}%")
    return sum(1 for _, ok, _ in checks if not ok)


def main() -> None:
    print("--- residual (one station) ---")
    bad = check_residual()
    print("\n--- whole propeller ---")
    bad += check_propeller()
    print(f"\n{'all checks passed' if bad == 0 else str(bad) + ' check(s) failed'}")


if __name__ == "__main__":
    main()
