"""Convert UIUC thrust/power coefficients into physical units.

The static data files give only dimensionless coefficients. Everything the
project is judged on -- target thrust in the input contract, thrust per shaft
watt in the success metric -- lives on the other side of this conversion, so
this is the module the whole success metric rests on.

    T = CT * rho * n^2 * D^4          thrust,      newtons
    P = CP * rho * n^3 * D^5          shaft power, watts

Per SCOPE.md, P is MECHANICAL SHAFT POWER at the propeller, not electrical
draw. Motor and ESC losses are out of scope for v1. This is also what the
UIUC data supports directly, since it comes from measured thrust and torque.

THREE UNIT TRAPS, in the order people hit them:

  1. n is REVOLUTIONS PER SECOND, not RPM. Using RPM directly inflates
     thrust by 60^2 = 3600x and power by 60^3 = 216000x. This is by far the
     most common error, and it is silent -- nothing raises, the numbers are
     just wrong.

  2. D is in METRES. parse_prop_name returns INCHES. Forgetting the 0.0254
     shrinks thrust by a factor of 0.0254^4, about 2.4 million.

  3. rho is in kg/m^3, which pins you to SI throughout. Do not mix in
     grams-force or inches partway.

A useful property: because T/P scales as (CT/CP) / (n*D), thrust per watt
should FALL as RPM or diameter rises. If your numbers do not show that, the
conversion is wrong somewhere -- see check 4 below.

    ./.venv/bin/python src/physics.py
"""

IN_TO_M = 0.0254
RHO_SEA_LEVEL = 1.225      # kg/m^3, ISA sea level -- SCOPE.md assumes this
GF_PER_N = 1000.0 / 9.80665  # newtons -> grams-force, for readable output


def thrust_N(ct, rpm, diameter_in, rho=RHO_SEA_LEVEL):
    n = rpm / 60.0
    d = diameter_in * IN_TO_M
    return ct*rho*(n**2) * (d**4)
    """Thrust in newtons, from the thrust coefficient.

        T = CT * rho * n^2 * D^4

    Args:
        ct: thrust coefficient (dimensionless), from a static data file.
        rpm: shaft speed in revolutions per MINUTE, as the file gives it.
        diameter_in: propeller diameter in INCHES, from parse_prop_name.
        rho: air density, kg/m^3.

    Returns:
        Thrust in newtons.

    Plain arithmetic here vectorizes over pandas Series and numpy arrays for
    free, so write it scalar-style and it will work on a whole column.

    TODO(ribhav): implement. Convert rpm -> rev/s and inches -> metres FIRST,
    on their own lines, then apply the formula. Doing the conversions inline
    inside one long expression is how trap 1 and trap 2 hide.
    """
    raise NotImplementedError


def shaft_power_W(cp, rpm, diameter_in, rho=RHO_SEA_LEVEL):
    n = rpm / 60.0
    d = diameter_in * IN_TO_M
    return cp * rho * (n**3) * (d**5)
    """Mechanical shaft power in watts, from the power coefficient.

        P = CP * rho * n^3 * D^5

    Note the exponents differ from thrust: n^3 and D^5, not n^2 and D^4.
    That difference is the entire reason efficiency falls with RPM.

    Args:
        cp: power coefficient (dimensionless), from a static data file.
        rpm: shaft speed in revolutions per MINUTE.
        diameter_in: propeller diameter in INCHES.
        rho: air density, kg/m^3.

    Returns:
        Shaft power in watts.

    TODO(ribhav): implement.
    """
    raise NotImplementedError


def thrust_per_watt(ct, cp, rpm, diameter_in, rho=RHO_SEA_LEVEL):
    return thrust_N(ct, rpm, diameter_in, rho)/shaft_power_W(cp, rpm, diameter_in, rho)
    """Thrust per shaft watt, in newtons per watt -- the success metric.

    This is what SCOPE.md judges the design on, and what the blade-count
    auto-selection maximizes.

    Returns:
        Thrust per shaft watt, N/W. Multiply by GF_PER_N for grams-force per
        watt, the unit the RC/drone world actually quotes (small propellers
        land around 7-12 gf/W).

    TODO(ribhav): implement. You can call the two functions above rather than
    rederiving; the rho, rpm and diameter arguments have to match.
    """
    raise NotImplementedError


# --- checks -----------------------------------------------------------------

def main() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from readers import find_static_runs, read_static

    checks = []

    # Reference point: APC 9x4.7 Slow Flyer at 6026 rpm, from
    # volume-1/data/apcsf_9x4.7_static_kt1032.txt -> CT=0.1192, CP=0.0484.
    # Expected values below are the hand-computed SI result.
    CT, CP, RPM, D_IN = 0.1192, 0.0484, 6026.0, 9.0

    t = thrust_N(CT, RPM, D_IN)
    p = shaft_power_W(CP, RPM, D_IN)
    e = thrust_per_watt(CT, CP, RPM, D_IN)

    checks.append(("thrust  ~= 4.02 N", abs(t - 4.022) < 0.01, f"{t:.4f} N ({t*GF_PER_N:.1f} gf)"))
    checks.append(("power   ~= 37.5 W", abs(p - 37.497) < 0.05, f"{p:.4f} W"))
    checks.append(("T/P     ~= 0.107 N/W", abs(e - 0.10727) < 0.001,
                   f"{e:.5f} N/W ({e*GF_PER_N:.2f} gf/W)"))

    # Plausibility: a 9" prop should land in the 7-12 gf/W band.
    checks.append(("T/P in 7-12 gf/W band", 7 < e * GF_PER_N < 12, f"{e*GF_PER_N:.2f} gf/W"))

    # Scaling law: thrust goes as n^2, so doubling rpm must quadruple thrust.
    t2 = thrust_N(CT, RPM * 2, D_IN)
    checks.append(("double rpm -> 4x thrust", abs(t2 / t - 4.0) < 1e-9, f"{t2/t:.6f}x"))

    # Scaling law: power goes as n^3, so doubling rpm must give 8x power.
    p2 = shaft_power_W(CP, RPM * 2, D_IN)
    checks.append(("double rpm -> 8x power", abs(p2 / p - 8.0) < 1e-9, f"{p2/p:.6f}x"))

    # Trap 1 detector: if rpm was used where rev/s belongs, thrust is 3600x high.
    checks.append(("not off by 60^2 (rpm/rev-s)", t < 100, f"{t:.2f} N"))
    # Trap 2 detector: if diameter stayed in inches, thrust is ~2.4e6x low.
    checks.append(("not off by 0.0254^4 (in/m)", t > 0.01, f"{t:.4f} N"))

    # Efficiency must fall as rpm rises, across a real measured sweep.
    run = next(r for r in find_static_runs() if r["prop_name"] == "apcsf_9x4.7")
    df = read_static(run["path"])
    eff = [thrust_per_watt(r.CT, r.CP, r.rpm, 9.0) for r in df.itertuples()]
    falling = all(a > b for a, b in zip(eff, eff[1:]))
    checks.append(("T/P falls monotonically with rpm", falling,
                   f"{eff[0]*GF_PER_N:.2f} -> {eff[-1]*GF_PER_N:.2f} gf/W "
                   f"over {df.rpm.iloc[0]:.0f}-{df.rpm.iloc[-1]:.0f} rpm"))

    ok = 0
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<34} {detail}")
        ok += passed
    print(f"\n{ok} passed, {len(checks)-ok} failed")


if __name__ == "__main__":
    main()
