"""Retraining stress: the lender refits the scorer on drifted data while the person is still working on a plan.

Idea from De Toni et al. (arXiv 2410.08007): a recommendation can become invalid if the model changes before the
plan is finished. We precompute a few retrained scorers once and reuse them for every applicant.
Each retrained scorer is fitted on a bootstrap resample of the labelled data in which the drifted (oot) cohort
makes up one third, calibrated on a bootstrap of validation plus drifted rows, and gets its own cost-based tau.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..model.scorer import Scorer

N_RETRAIN = 5
DRIFT_SHARE = 1 / 3


def _boot(pool, n, rng):
    return pool.iloc[rng.integers(0, len(pool), n)]


def build_retrained(people, feats, n_models=N_RETRAIN, seed=C.SEED):
    """people: DataFrame with split and y_true; feats: DataFrame indexed by id. Returns fitted Scorers with .tau."""
    df = people[["split", "y_true"]].join(feats, how="inner")
    base = df[df.split.isin(["train", "val"])]
    drift = df[df.split == "oot"]
    if len(drift) < 50:  # fixtures without an oot cohort: reuse validation rows as the drifted pool
        drift = df[df.split == "val"]
    val = df[df.split == "val"]
    n_fit = int((df.split == "train").sum()) or len(base)
    n_cal = max(len(val), 200)
    out = []
    for i in range(n_models):
        rng = np.random.default_rng([seed, i])
        k = int(n_fit * DRIFT_SHARE)
        fit = pd.concat([_boot(base, n_fit - k, rng), _boot(drift, k, rng)])
        cal = pd.concat([_boot(val, n_cal // 2, rng), _boot(drift, n_cal - n_cal // 2, rng)])
        if fit.y_true.nunique() < 2 or cal.y_true.nunique() < 2:
            continue
        s = Scorer().fit(fit, fit.y_true.values, cal, cal.y_true.values)
        s.set_tau_cost(cal, cal.y_true.values)
        out.append(s)
    return out
