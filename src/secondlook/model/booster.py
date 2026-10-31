"""Optional monotone-constrained gradient-boosted scorer, for comparison with the logistic scorecard."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .. import config
from ..contracts import FEATURE_DIRECTION
from .scorer import Scorer


class Booster:
    """y = 1 means default. Same interface as Scorer where it matters: predict_proba, decide, set_tau_cost."""

    def __init__(self, features, n_estimators=200, max_depth=3, learning_rate=0.05, seed=config.SEED):
        self.features = list(features)
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2, tree_method="hist",
                           monotone_constraints="(" + ",".join(str(FEATURE_DIRECTION[f]) for f in self.features) + ")")

    def fit(self, X, y, X_cal, y_cal):
        self.xgb_ = XGBClassifier(**self.params).fit(X[self.features].values, y)
        self.cal_ = LogisticRegression(C=1e6, max_iter=1000).fit(self.logit(X_cal).reshape(-1, 1), y_cal)  # Platt
        return self

    def logit(self, X):
        return self.xgb_.predict(X[self.features].values, output_margin=True)

    def predict_proba(self, X):
        return self.cal_.predict_proba(self.logit(X).reshape(-1, 1))[:, 1]

    def decide(self, X):
        return (self.predict_proba(X) <= self.tau).astype(int)

    def set_tau_cost(self, X, y, c_default=config.COST_DEFAULT, c_lost=config.COST_LOST):
        return Scorer.set_tau_cost(self, X, y, c_default, c_lost)   # same grid search as the scorecard

    def shap_importance(self, X):
        """Mean |SHAP| per feature (raw margin units)."""
        import shap
        v = shap.TreeExplainer(self.xgb_).shap_values(X[self.features].values)
        return np.abs(v).mean(axis=0)
