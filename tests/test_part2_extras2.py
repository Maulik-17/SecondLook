import numpy as np
import pandas as pd
import pytest

from secondlook import config as C
from secondlook.audit.intersections import scan
from secondlook.contracts import GROUPS
from secondlook.uncertainty.gamma import gamma_sensitivity, perturb_weights


def _cal_te(N=3000, seed=0):
    r = np.random.default_rng(seed)

    def make(n):
        p = np.clip(r.beta(1, 6, n), 0.01, 0.99)
        return pd.DataFrame({"p_default": p, "y_true": (r.random(n) < p).astype(int), "thin_file": r.integers(0, 2, n),
                             "gender": r.choice(["F", "M"], n)})
    cal, te = make(N), make(N)
    cal["y_obs"] = cal.y_true
    return cal, te, r.uniform(1, 5, N)


def test_gamma_one_matches_plain_fit_and_perturb_clips():
    from secondlook.uncertainty.conformal import ConformalModel
    cal, te, w = _cal_te()
    out = gamma_sensitivity(cal, w, te, 0.1, gammas=(1.0, 2.0))
    m = ConformalModel().fit(cal, w, 0.1)
    res = m.predict_set(te.p_default.values, te)
    cov = np.mean([int(y) in x["set"] for y, x in zip(te.y_true, res)])
    assert abs(out["rows"][0]["overall_coverage"] - cov) < 1e-12
    assert len(out["rows"]) == 2 and "not a guarantee" in out["note"]
    pw = perturb_weights(np.linspace(0, 1, 10), np.full(10, C.W_MAX), 3.0, "hard_up")
    assert pw.max() <= C.W_MAX
    with pytest.raises(ValueError):
        gamma_sensitivity(cal, w, te, 0.1, gammas=(0.5,))


def test_gamma_worst_direction_never_raises_coverage_above_gamma_one_much():
    cal, te, w = _cal_te(seed=3)
    out = gamma_sensitivity(cal, w, te, 0.1, gammas=(1.0, 3.0))
    assert out["rows"][1]["overall_coverage"] <= out["rows"][0]["overall_coverage"] + 0.02


def test_intersection_scan_skips_small_and_counts():
    r = np.random.default_rng(1)
    n = 600
    te = pd.DataFrame({a: r.choice(GROUPS[a], n) for a in GROUPS})
    te.loc[:4, ["gender", "region"]] = ["F", "R6"]          # make sure some cells are tiny
    eff = r.exponential(1.0, n)
    rejected = r.random(n) < 0.5
    rg = rejected & (r.random(n) < 0.5)
    out = scan(te, eff, rg, rejected, attrs=["gender", "region", "thin_file"], B=20, top=3)
    assert out["n_cells_examined"] + out["n_cells_skipped_small"] == 2 * 6 + 2 * 2 + 6 * 2
    assert all(x["n"] >= C.MIN_GROUP_N for x in out["rows"])
    assert len(out["worst"]) == 3 and out["worst"][0]["burden"][1] >= out["worst"][1]["burden"][1]
    assert str(out["n_cells_examined"]) in out["caution"]
    lo, est, hi = out["rows"][0]["burden"]
    assert lo <= est <= hi
    assert 0 <= out["rows"][0]["approval"][1] <= 1


def test_intersection_burden_matches_manual():
    n = 200
    te = pd.DataFrame({"gender": ["F"] * 100 + ["M"] * 100, "thin_file": [0, 1] * 100})
    eff = np.arange(n, dtype=float)
    rejected = np.ones(n, bool)
    rg = np.ones(n, bool)
    out = scan(te, eff, rg, rejected, attrs=["gender", "thin_file"], B=0)
    row = next(x for x in out["rows"] if x["l1"] == "F" and x["l2"] == 0)
    assert abs(row["burden"][1] - eff[:100][::2].mean()) < 1e-12
