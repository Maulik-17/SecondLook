import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from secondlook import config as C, contracts as K
from secondlook.sim.generate import generate


@pytest.fixture(scope="module")
def d():
    return generate(6000, 1)


def test_reproducible():
    a, b = generate(800, 5), generate(800, 5)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    c = generate(800, 6)
    assert not a["people"].r_true.equals(c["people"].r_true)


def test_schema_and_ids(d):
    for n in K.TABLES:
        if n in d:
            K.check(n, d[n])
    assert d["people"].id.is_unique and d["features"].id.is_unique
    assert list(d["features"].columns) == ["id"] + K.FEATURES
    assert list(d["latents"].columns) == K.LATENTS_COLUMNS
    assert set(d["monthly"].month) == set(range(1, 37))
    bad = {"gender", "region", "urban_rural", "occupation", "mu_income", "r_true", "y_true"}
    assert not bad & set(d["features"].columns)


def test_base_rate_and_oracle(d):
    p = d["people"]
    assert 0.09 <= p.y_true.mean() <= 0.13
    assert 0.74 <= roc_auc_score(p.y_true, p.r_true) <= 0.82


def test_scorer_auc_in_range(d):
    p, F = d["people"], d["features"].drop(columns="id")
    X = ((F - F.mean()) / F.std()).values
    tr, te = (p.split == "train").values, (p.split == "test").values
    auc = roc_auc_score(p.y_true[te], LogisticRegression(max_iter=1000).fit(X[tr], p.y_true[tr]).predict_proba(X[te])[:, 1])
    assert 0.68 <= auc <= 0.78


def test_temporal_rule(d):
    # features use only the history window: the stored months 1..24 reproduce the features for thick-file people
    m = d["monthly"]
    assert m[(m.month > 24)].groupby("id").size().eq(12).all()
    p, F = d["people"], d["features"]
    thick = p.id[p.thin_file == 0].values[:20]
    from secondlook.model.features import cash_features
    for i in thick:
        h = m[(m.id == i) & (m.month <= 24)].sort_values("month")
        f = cash_features(h[["inflow"]].values.T, h[["outflow"]].values.T, h[["balance"]].values.T, [1.0])
        assert np.isclose(f["mean_inflow"][0], F.loc[F.id == i, "mean_inflow"].iloc[0])


def test_lender_history(d):
    lh = d["lender_history"]
    assert (lh.y_obs.isna() == (lh.observed == 0)).all()
    assert ((lh.propensity > 0) & (lh.propensity <= 1)).all()


def test_bias_flags_control(monkeypatch):
    on = generate(4000, 2)
    auc_on = roc_auc_score(on["people"].gender == "F", on["features"].night_active_share)
    assert 0.58 <= auc_on <= 0.72
    monkeypatch.setattr(C, "BIAS_FLAGS", False)
    off = generate(4000, 2)
    auc_off = roc_auc_score(off["people"].gender == "F", off["features"].night_active_share)
    assert 0.45 <= auc_off <= 0.55


def test_null_world_unequal_base_rates_no_bias():
    z = generate(6000, 3, "null")
    p, F = z["people"], z["features"]
    br = p.groupby("urban_rural").y_true.mean()
    assert br["rural"] > br["urban"] + 0.02                       # unequal base rates
    assert 0.45 <= roc_auc_score(p.gender == "F", F.night_active_share) <= 0.55   # no planted gender proxy
    lh = z["lender_history"]
    assert (lh.y_obs.isna() == (lh.observed == 0)).all()


def test_oot_split(d):
    o = generate(500, 9, "oot")
    assert (o["people"].split == "oot").all()
