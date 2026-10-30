import numpy as np
import pytest

from secondlook import contracts as K
from secondlook.model import explain
from secondlook.model.scorer import Scorer
from secondlook.sim.generate import generate


@pytest.fixture(scope="module")
def fit():
    d = generate(6000, 11)
    df = d["people"].merge(d["features"], on="id")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    s = Scorer().fit(tr, tr.y_true.values, va, va.y_true.values)
    s.set_tau_cost(va, va.y_true.values)
    return s, df


def test_threshold_consistent(fit):
    s, df = fit
    thr = s.logit_threshold()
    assert ((s.logit(df) <= thr) == (s.predict_proba(df) <= s.tau)).mean() > 0.999


def test_linear_params_exact(fit):
    s, df = fit
    lp = s.linear_params()
    assert np.allclose(df[lp["features"]].values @ lp["w"] + lp["b"], s.logit(df), atol=1e-6)


def test_reasons_sum_to_logit(fit):
    s, df = fit
    row = df.iloc[[3]]
    lp = s.linear_params()
    mean = dict(zip(s.features, s.mean_))
    total = sum(w * (row[f].iloc[0] - mean[f]) for f, w in zip(lp["features"], lp["w"]))
    const = lp["b"] + float(np.dot(lp["w"], list(mean.values())))
    assert abs(total + const - s.logit(row)[0]) < 1e-6
    r = explain.reasons(s, row)
    assert len(r) == 5 and all(x["feature"] not in explain.SKIP for x in r)


def test_text_complete():
    for f in K.FEATURES:
        if f in explain.SKIP:
            continue
        assert (f, "raises risk") in explain.TEXT and (f, "lowers risk") in explain.TEXT


def test_sign_constrained_training(fit):
    # emulate scripts/train.py loop
    _, df = fit
    tr, va = df[df.split == "train"], df[df.split == "val"]
    feats = list(K.FEATURES)
    while True:
        s = Scorer(feats).fit(tr, tr.y_true.values, va, va.y_true.values)
        w = dict(zip(s.features, s.lr_.coef_[0]))
        bad = [f for f in feats if K.FEATURE_DIRECTION[f] * w[f] < 0]
        if not bad:
            break
        feats.remove(max(bad, key=lambda f: abs(w[f])))
    assert all(K.FEATURE_DIRECTION[f] * w[f] >= 0 for f in feats)
