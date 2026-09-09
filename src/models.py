"""Inverse-design models: features -> blade geometry.

Two ways to predict a blade, and the difference matters more than the
algorithm choice.

RAW: predict all 36 station values directly.
    Flexible -- it can express any blade, including ones outside the
    generator's parameterization. But nothing couples the stations, so the
    model can emit a chord distribution with a kink in it. project-plan.md
    flags exactly this: per-station prediction "can produce jagged,
    physically-invalid blade shapes".

PARAM: predict the five generator parameters, then rebuild the blade.
    Pitch ratio, solidity, washout, and two chord-shape components. Only five
    numbers instead of 36, and every output is smooth and manufacturable BY
    CONSTRUCTION -- the reconstruction cannot produce a kink. The cost is that
    it can only express blades inside that parameterization.

Comparing them is the experiment project-plan.md asks for, with the smoothness
guaranteed structurally rather than hoped for.
"""

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import FEATURES, PARAMS, TARGETS
from generate_synthetic import chord_shape_basis, make_geometry

# Untuned, and deliberately modest. With only three input features a tree
# ensemble has little room to overfit the inputs; depth and count here are
# about resolving the target manifold, not the feature space.
XGB_KW = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    tree_method="hist",
    # One tree predicting all outputs jointly, rather than 36 independent
    # models. Outputs share structure -- neighbouring stations are strongly
    # correlated -- so joint trees both train faster and keep predictions
    # more coherent across the blade.
    multi_strategy="multi_output_tree",
    random_state=0,
)


class XGBRaw:
    """Predict the 36 station values directly."""

    def fit(self, X, y):
        self.m_ = xgb.XGBRegressor(**XGB_KW).fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.m_.predict(np.asarray(X))


class XGBParam:
    """Predict the five generator parameters, then rebuild the geometry.

    The reconstruction is the same code the training data was generated with,
    so a predicted parameter vector maps to a blade the same way a sampled one
    did. Diameter cancels out of the normalized outputs -- twist is
    arctan(pitch_ratio / (pi * r/R)) and chord is c/R -- so no dimensional
    input is needed to rebuild.
    """

    def fit(self, X, y, params=None):
        if params is None:
            raise ValueError("XGBParam needs the generator parameters as targets")
        self.mean_shape_, self.components_ = chord_shape_basis()
        self.m_ = xgb.XGBRegressor(**XGB_KW).fit(np.asarray(X), np.asarray(params))
        return self

    def predict(self, X):
        X = np.asarray(X)
        p = self.m_.predict(X)
        blade_count = X[:, FEATURES.index("blade_count")]

        out = np.zeros((len(X), len(TARGETS)))
        for i in range(len(X)):
            pitch_ratio, solidity, washout, c1, c2 = p[i]
            c_R, beta = make_geometry(
                pitch_ratio=max(pitch_ratio, 0.05),
                diameter_in=10.0,              # cancels; any positive value
                solidity=max(solidity, 0.01),
                washout_deg=washout,
                shape_coeffs=np.array([c1, c2]),
                blade_count=max(blade_count[i], 1.0),
                mean_shape=self.mean_shape_,
                components=self.components_,
            )
            out[i, :len(c_R)] = c_R
            out[i, len(c_R):] = beta
        return out
