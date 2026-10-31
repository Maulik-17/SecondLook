"""Sensitivity of weighted conformal coverage to unobserved confounding of the lender's exploration/approval.

Rambachan-style idea: the inverse-propensity weights assume the outcome does not depend on selection once the
features are known. If an unobserved factor shifted the true selection odds by at most a factor Gamma, each true
weight could differ from the one we used by up to Gamma in either direction. For each Gamma the two adversarial
patterns are tried and the worse coverage is reported:
  "hard_up":   calibration rows with above-median nonconformity score (hard cases) get w * Gamma, easy rows w / Gamma
  "hard_down": the opposite
Weights are clipped at config.W_MAX. (A uniform factor on all weights would change nothing, since the weighted
quantile is scale-free, which is why the factor is applied in opposite directions to hard and easy rows.)

This is a SENSITIVITY ANALYSIS, NOT A GUARANTEE: it shows how far coverage could move under a stated amount of
hidden confounding, not that coverage holds or that Gamma is known. Gamma = 1 reproduces the ordinary weighted fit.
"""
import numpy as np

from .. import config as C
from ..contracts import GROUPS
from .conformal import ConformalModel

GAMMAS = (1.0, 1.25, 1.5, 2.0, 3.0)
NOTE = ("Sensitivity analysis, not a guarantee: coverage is recomputed after the exploration weights are pushed "
        "by up to a factor Gamma (hard cases up and easy cases down, or the reverse), clipped at W_MAX. "
        "It shows how much hidden confounding of approval it takes to move coverage.")


def perturb_weights(s, w, gamma, direction, cells_idx=None):
    """Scale weights by gamma^(+-1) according to whether the score is above the (per cell) median score."""
    s, w = np.asarray(s, float), np.asarray(w, float)
    hard = np.zeros(len(s), bool)
    parts = cells_idx if cells_idx is not None else [np.arange(len(s))]
    for idx in parts:
        if len(idx):
            hard[idx] = s[idx] > np.median(s[idx])
    sign = 1.0 if direction == "hard_up" else -1.0
    f = np.where(hard, gamma ** sign, gamma ** -sign)
    return np.minimum(w * f, C.W_MAX)


def _eval(model, te):
    res = model.predict_set(te.p_default.values, te)
    abst = np.array([r["abstain"] for r in res])
    cover = np.array([int(y) in r["set"] for y, r in zip(te.y_true.values, res)])
    return cover, abst


def gamma_sensitivity(d, weights, te, alpha, gammas=GAMMAS, cells=None):
    """d: observed calibration rows (p_default, y_obs, cell columns); weights: IPW weights (same order);
    te: fully labelled evaluation rows (p_default, y_true, group columns). Returns a dict with one entry per Gamma:
    overall and per-group coverage / referral under the worse of the two directions (lower coverage)."""
    cells = list(cells or ["thin_file", "gender"])
    d = d.reset_index(drop=True)
    w0 = np.minimum(np.asarray(weights, float), C.W_MAX)
    s = np.where(d.y_obs.values == 1, 1 - d.p_default.values, d.p_default.values)
    parts = list(d.groupby(cells).indices.values())
    rows = []
    for g in gammas:
        if g < 1:
            raise ValueError("gamma must be >= 1")
        best = None
        for direction in ("hard_up", "hard_down"):
            w = w0 if g == 1 else perturb_weights(s, w0, g, direction, parts)
            cover, abst = _eval(ConformalModel().fit(d, w, alpha, cells), te)
            if best is None or cover.mean() < best[0].mean():
                best = (cover, abst, direction)
        cover, abst, direction = best
        groups = []
        for a, levels in GROUPS.items():
            if a not in te.columns:
                continue
            for l in levels:
                mk = (te[a] == l).values
                if mk.sum() >= C.MIN_GROUP_N:
                    groups.append({"attribute": a, "level": l.item() if hasattr(l, "item") else l, "n": int(mk.sum()),
                                   "coverage": float(cover[mk].mean()), "referral": float(abst[mk].mean())})
        rows.append({"gamma": float(g), "worst_direction": direction, "overall_coverage": float(cover.mean()),
                     "overall_referral": float(abst.mean()),
                     "min_group_coverage": min([x["coverage"] for x in groups] or [float("nan")]), "groups": groups})
    return {"gammas": [float(g) for g in gammas], "rows": rows, "w_max": C.W_MAX, "note": NOTE}
