"""Reference scorer. Part 1 owns it and may replace the internals; the public interface below must not change."""
import joblib
import numpy as np
from scipy.special import expit, logit as _logit
from sklearn.linear_model import LogisticRegression

from .. import config
from ..contracts import FEATURES


class Scorer:
    """y = 1 means default. p_default is calibrated; logit is the raw (pre-calibration) linear score."""

    def __init__(self, features=FEATURES):
        self.features = list(features)

    def fit(self, X, y, X_cal, y_cal):
        self.mean_, self.scale_ = X[self.features].mean().values, X[self.features].std().values + 1e-9
        self.lr_ = LogisticRegression(C=1.0, max_iter=1000).fit(self._z(X), y)
        self.cal_ = LogisticRegression(C=1e6, max_iter=1000).fit(self.logit(X_cal).reshape(-1, 1), y_cal)  # Platt
        return self

    def _z(self, X):
        return (X[self.features].values - self.mean_) / self.scale_

    def logit(self, X) -> np.ndarray:
        return self.lr_.decision_function(self._z(X))

    def predict_proba(self, X) -> np.ndarray:
        """Calibrated p_default in [0, 1]."""
        return self.cal_.predict_proba(self.logit(X).reshape(-1, 1))[:, 1]

    def linear_params(self) -> dict:
        """Raw-feature-unit weights so that logit(x) = w @ x + b exactly (used by the burden audit)."""
        w = self.lr_.coef_[0] / self.scale_
        return {"features": self.features, "w": w, "b": float(self.lr_.intercept_[0] - w @ self.mean_)}

    def logit_threshold(self) -> float:
        """The raw-logit value s* whose calibrated p_default equals self.tau."""
        a, b = self.cal_.coef_[0, 0], self.cal_.intercept_[0]
        return float((_logit(self.tau) - b) / a)

    def decide(self, X) -> np.ndarray:
        """1 = approve (p_default <= tau)."""
        return (self.predict_proba(X) <= self.tau).astype(int)

    def set_tau_cost(self, X, y, c_default=config.COST_DEFAULT, c_lost=config.COST_LOST):
        p, grid = self.predict_proba(X), np.linspace(0.01, 0.99, 197)
        cost = [c_default * ((p <= t) & (y == 1)).sum() + c_lost * ((p > t) & (y == 0)).sum() for t in grid]
        self.tau = float(grid[int(np.argmin(cost))])
        return self.tau


def save_model(s: Scorer, path=None):
    config.ART_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(s, path or config.ART_DIR / "model.joblib")


def load_model(path=None) -> Scorer:
    return joblib.load(path or config.ART_DIR / "model.joblib")
