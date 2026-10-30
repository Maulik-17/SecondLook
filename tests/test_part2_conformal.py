import numpy as np
import pandas as pd
import pytest

from secondlook import config as C
from secondlook.uncertainty.conformal import ConformalModel, wquantile, ess, run_uncertainty
from secondlook.uncertainty.exploration import redraw


def test_wquantile():
    assert wquantile(np.array([0.1, 0.2, 0.3]), np.ones(3), 0.5, 1.0) == 0.2
    assert np.isinf(wquantile(np.array([0.1]), np.ones(1), 0.99, 1.0))


def test_ess_equals_n_for_unit_weights():
    assert abs(ess(np.ones(37)) - 37) < 1e-9


def test_redraw_zero_raises():
    df = pd.DataFrame({"approved_legacy": [1, 0, 0]})
    with pytest.raises(ValueError):
        redraw(df, 0)
    e, o, p = redraw(df, 0, allow_zero=True)
    assert not e.any() and list(p) == [1, 0, 0]


def test_predict_set_abstain_and_singleton():
    m = ConformalModel()
    m.cells, m.q, m.q_global = ["thin_file", "gender"], {}, 0.4
    g = pd.DataFrame({"thin_file": [0, 0], "gender": ["F", "F"]})
    r = m.predict_set(np.array([0.5, 0.02]), g)
    assert r[0]["abstain"]
    assert r[1]["set"] == [0] and not r[1]["abstain"]


def _toy(N, seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=N)
    p = 1 / (1 + np.exp(-(x - 2)))
    y = (r.random(N) < p).astype(int)
    appr = (r.random(N) < 1 / (1 + np.exp(2 * (x - 0.5)))).astype(int)   # selection depends on x only
    return pd.DataFrame({"p_default": p, "y_true": y, "approved_legacy": appr, "thin_file": r.integers(0, 2, N),
                         "gender": r.choice(["F", "M"], N)})


def test_weighted_coverage_on_fresh_population():
    alpha, eps = 0.1, 0.05
    cal, te = _toy(40000, 1), _toy(40000, 2)
    expl, obs, prop = redraw(cal, eps, seed=3)
    d = cal[obs].copy()
    d["y_obs"] = d.y_true
    w = np.minimum(1 / prop[obs], C.W_MAX)
    m = ConformalModel().fit(d, w, alpha)
    res = m.predict_set(te.p_default.values, te)
    cov = np.mean([int(y) in r["set"] for y, r in zip(te.y_true, res)])
    assert abs(cov - (1 - alpha)) < 0.03


def test_run_uncertainty_shape():
    from secondlook.audit.data import load
    try:
        df = load()
    except Exception:
        pytest.skip("no data")
    out = run_uncertainty(0.03, 0.1, df=df, write=False)
    assert set(out["variants"]) == {"unweighted_accepts", "unweighted_explored", "weighted"}
    v = out["variants"]["weighted"]
    assert v["groups"] and {"attribute", "level", "n", "coverage", "referral", "ess"} <= set(v["groups"][0])
