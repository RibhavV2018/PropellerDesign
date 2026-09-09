"""Scoring harness for Phase 2 inverse-design models.

TWO FAMILIES OF METRIC, AND THEY DISAGREE ON PURPOSE

Geometric  -- how close is the predicted blade to the real one, in chord and
              twist? Standard, cheap, and the WRONG question on its own. The
              inverse map is not unique: several genuinely different blades
              all produce the target thrust, so penalizing a good blade for
              looking unlike one particular reference punishes correct
              answers. Reported because reviewers expect it, and because a
              large gap between the two families is itself a finding.

Closed-loop -- run the predicted geometry back through BEMT and ask what it
              actually does. Does it hit the target thrust? At what CT/CP?
              This is what SCOPE.md judges, and it is the question a
              propeller designer would ask.

ISOLATING THE MODEL FROM THE SOLVER

The solver carries a known ~12% bias against measurement, so comparing
BEMT(predicted blade) against MEASURED performance would charge the model for
the solver's error. Both references are therefore reported:

    vs BEMT(real geometry)  -- the ML model's own error, solver bias cancels
    vs measurement          -- end-to-end error of the whole pipeline

The first is the fair test of the model. The second is what a user gets.

    ./.venv/bin/python src/evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bemt import low_re_airfoil, solve_propeller
from dataset import B_COLS, C_COLS, FEATURES, STATIONS, TARGETS, add_features, load_real, load_synthetic

REAL_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "uiuc_hover.csv"

N_CHORD = len(C_COLS)


# --- metrics ----------------------------------------------------------------

def geometric_metrics(y_true, y_pred) -> dict:
    """Chord and twist error, reported separately -- they have different units.

    Chord is c/R (dimensionless, ~0.1-0.3); twist is degrees (~5-45). A single
    blended number across both would be dominated by whichever happens to have
    the larger numeric scale and would mean nothing.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    c_t, c_p = yt[:, :N_CHORD], yp[:, :N_CHORD]
    b_t, b_p = yt[:, N_CHORD:], yp[:, N_CHORD:]

    def r2(a, b):
        ss_res = ((a - b) ** 2).sum()
        ss_tot = ((a - a.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    return {
        "chord_MAE": float(np.abs(c_t - c_p).mean()),
        "chord_RMSE": float(np.sqrt(((c_t - c_p) ** 2).mean())),
        "chord_R2": float(r2(c_t, c_p)),
        "twist_MAE_deg": float(np.abs(b_t - b_p).mean()),
        "twist_RMSE_deg": float(np.sqrt(((b_t - b_p) ** 2).mean())),
        "twist_R2": float(r2(b_t, b_p)),
    }


def run_bemt(geometry_row, diameter_in, rpm, blade_count):
    """Solve one blade. Returns (CT, CP) or (nan, nan) if it fails."""
    c_R = list(geometry_row[:N_CHORD])
    beta = list(geometry_row[N_CHORD:])
    try:
        out = solve_propeller(STATIONS, c_R, beta, diameter_in, rpm,
                              int(blade_count), airfoil=low_re_airfoil)
        return out["CT"], out["CP"]
    except Exception:
        return np.nan, np.nan


def closed_loop_metrics(meta, y_true, y_pred, ref_cache={}) -> dict:
    """Run predicted blades through BEMT and score what they actually do.

    Args:
        meta: DataFrame with diameter_in, rpm, blade_count, CT (measured).
        y_true: the real measured geometry, for the solver-bias-free reference.
        y_pred: predicted geometry.

    Returns:
        thrust_error  -- does the blade hit the requested CT?
        ctcp_ratio    -- predicted blade's CT/CP over the real blade's, both
                         through the same solver, so solver bias cancels.
                         Above 1.0 means the model designed a BETTER blade
                         than the real propeller, which is allowed and is the
                         point of the exercise.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)

    # BEMT on the real geometry -- cached, since it is the same for every model
    key = id(y_true)
    if key not in ref_cache:
        ref_cache[key] = np.array([
            run_bemt(yt[i], m.diameter_in, m.rpm, m.blade_count)
            for i, m in enumerate(meta.itertuples())
        ])
    ref = ref_cache[key]

    got = np.array([
        run_bemt(yp[i], m.diameter_in, m.rpm, m.blade_count)
        for i, m in enumerate(meta.itertuples())
    ])

    ok = np.isfinite(got).all(1) & np.isfinite(ref).all(1) & (ref[:, 1] > 0) & (got[:, 1] > 0)
    ct_target = meta.CT.values

    thrust_err = (got[ok, 0] / ct_target[ok] - 1) * 100
    ctcp_pred = got[ok, 0] / got[ok, 1]
    ctcp_ref = ref[ok, 0] / ref[ok, 1]

    return {
        "solved_pct": float(ok.mean() * 100),
        "thrust_err_median_pct": float(np.median(thrust_err)),
        "thrust_err_MAE_pct": float(np.median(np.abs(thrust_err))),
        "ctcp_vs_real": float(np.median(ctcp_pred / ctcp_ref)),
        "ctcp_pred_median": float(np.median(ctcp_pred)),
        "ctcp_real_median": float(np.median(ctcp_ref)),
    }


# --- baselines --------------------------------------------------------------

class MeanBlade:
    """Predict the training-set mean blade, ignoring the inputs entirely.

    The floor. Any model that cannot beat this has learned nothing, and it is
    a more honest reference than R^2 against zero.
    """

    def fit(self, X, y):
        self.mean_ = np.asarray(y).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


class ClassicalDesign:
    """Textbook hover blade design -- physics, no fitting.

    The baseline a reviewer will want to see beaten, because it is what an
    engineer would do without any machine learning:

      Solidity from blade-element theory. For uniform inflow and constant
      section lift, CT = sigma*Cl/6, so sigma = 6*CT/Cl at a design Cl.

      Twist from ideal hover twist, anchored at 0.75R. Momentum theory gives
      the inflow ratio lambda = sqrt(CT/2); the section angle is that inflow
      plus a design angle of attack; and constant geometric pitch propagates
      it along the blade -- the same 1/r law the generator uses.

    Chord shape is uniform, which is where it is weakest: real blades taper.

    CONVENTION TRAP: both of those textbook relations are written in the
    HELICOPTER thrust coefficient,

        CT_heli = T / (rho * pi*R^2 * (Omega*R)^2)

    while this project uses the PROPELLER convention throughout,

        CT_prop = T / (rho * n^2 * D^4)

    They differ by 4/pi^3 = 0.129. Feeding CT_prop straight into sigma =
    6*CT/Cl gives solidity around 1.4 against a real range of 0.065-0.186 --
    a blade roughly ten times too fat, and a baseline that fails so badly it
    looks like the method is wrong rather than the units.
    """

    CT_HELI_PER_PROP = 4.0 / np.pi ** 3     # 0.129
    CL_DESIGN = 0.5
    ALPHA_DESIGN_DEG = 5.0

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = pd.DataFrame(X, columns=FEATURES)
        st = np.array(STATIONS)
        out = np.zeros((len(X), len(TARGETS)))

        for i, (ct_prop, B, _) in enumerate(X.itertuples(index=False)):
            ct = max(ct_prop, 1e-6) * self.CT_HELI_PER_PROP   # -> helicopter convention
            sigma = 6.0 * ct / self.CL_DESIGN
            # constant chord hitting that solidity: sigma = B*c/(pi*R)
            c_R = np.full_like(st, sigma * np.pi / B)

            lam = np.sqrt(ct / 2.0)
            phi75 = lam / 0.75
            beta75 = phi75 + np.radians(self.ALPHA_DESIGN_DEG)
            P_over_D = 0.75 * np.pi * np.tan(beta75)
            beta = np.degrees(np.arctan(P_over_D / (2.0 * np.pi * (st / 2.0))))

            out[i, :N_CHORD] = c_R
            out[i, N_CHORD:] = beta
        return out


# --- harness ----------------------------------------------------------------

def evaluate(models, subsample=None, seed=0) -> pd.DataFrame:
    """Train each model on synthetic data, score it on real propellers.

    Training is synthetic and testing is measured, deliberately. A model
    trained on BEMT output and graded against BEMT output would only prove it
    can imitate the solver.
    """
    Xs, ys, ps = load_synthetic()
    Xr, yr, groups = load_real()
    meta = add_features(pd.read_csv(REAL_CSV))

    if subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xr), size=min(subsample, len(Xr)), replace=False)
        Xr, yr, meta = Xr.iloc[idx], yr.iloc[idx], meta.iloc[idx]

    rows = []
    for name, model in models.items():
        if type(model).__name__ == "XGBParam":
            model.fit(Xs.values, ys.values, params=ps.values)
        else:
            model.fit(Xs.values, ys.values)
        pred = model.predict(Xr.values)
        rec = {"model": name}
        rec.update(geometric_metrics(yr.values, pred))
        rec.update(closed_loop_metrics(meta, yr.values, pred))
        rows.append(rec)
        print(f"  scored {name}")
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    from sklearn.linear_model import LinearRegression
    from models import XGBParam, XGBRaw

    models = {
        "mean_blade": MeanBlade(),
        "classical": ClassicalDesign(),
        "linear": LinearRegression(),
        "xgb_raw": XGBRaw(),
        "xgb_param": XGBParam(),
    }
    print(f"evaluating {len(models)} baselines on real propellers...")
    res = evaluate(models)

    print("\n--- geometric (how close to the real blade) ---")
    print(res[["chord_MAE", "chord_R2", "twist_MAE_deg", "twist_R2"]].round(3).to_string())

    print("\n--- closed-loop (what the blade actually does) ---")
    print(res[["solved_pct", "thrust_err_median_pct", "thrust_err_MAE_pct",
               "ctcp_vs_real"]].round(2).to_string())
    print("\n  thrust_err : does the predicted blade hit the requested CT")
    print("  ctcp_vs_real: predicted blade's CT/CP over the real blade's,")
    print("                both through the same solver.  > 1.0 = better than real.")


if __name__ == "__main__":
    main()
