import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from secondlook import config, contracts as K
from secondlook.model.booster import Booster

FEATS = ["median_inflow", "spike_months", "low_balance_days", "expense_to_income", "emi_to_income", "months_observed"]


def make(n=2000, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f: r.normal(size=n) for f in FEATS})
    z = sum(K.FEATURE_DIRECTION[f] * X[f] for f in FEATS) * 0.6 - 1.5 + r.normal(size=n)
    return X, (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)


def test_monotone_constraints_hold():
    X, y = make()
    b = Booster(FEATS, n_estimators=60).fit(X.iloc[:1200], y[:1200], X.iloc[1200:], y[1200:])
    base = X.iloc[:200]
    grid = np.linspace(-3, 3, 25)
    for f in FEATS:
        d = K.FEATURE_DIRECTION[f]
        if d == 0:
            continue
        curve = np.array([b.predict_proba(base.assign(**{f: v})).mean() for v in grid])
        assert (d * np.diff(curve) >= -1e-9).all(), f


def test_tau_and_decide():
    X, y = make()
    b = Booster(FEATS, n_estimators=30).fit(X.iloc[:1200], y[:1200], X.iloc[1200:], y[1200:])
    tau = b.set_tau_cost(X.iloc[1200:], y[1200:])
    assert 0.01 <= tau <= 0.99 and set(np.unique(b.decide(X))) <= {0, 1}


def test_comparison_json_schema(tmp_path, monkeypatch):
    import run_booster
    X, y = make(2000)
    r = np.random.default_rng(1)
    ppl = pd.DataFrame({"id": np.arange(len(X)), "y_true": y,
                        "split": np.array(["train", "val", "test", "oot"])[np.arange(len(X)) % 4]})
    for g, vals in K.GROUPS.items():
        ppl[g] = r.choice(vals, len(X))
    ppl.to_parquet(tmp_path / "people.parquet")
    X.assign(id=np.arange(len(X))).to_parquet(tmp_path / "features.parquet")
    (tmp_path / "metrics.json").write_text(json.dumps({"kept_features": FEATS}))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ART_DIR", tmp_path)
    monkeypatch.setattr(config, "MIN_GROUP_N", 5)
    run_booster.main(n_boot=2)
    out = json.loads((tmp_path / "booster_comparison.json").read_text())
    assert out["disclaimer"] and set(out) >= {"logistic", "booster", "stability", "tau"}
    for m in ["logistic", "booster"]:
        for s in ["test", "oot"]:
            assert len(out[m][s]["auc"]) == 3 and set(out[m][s]) >= {"brier", "ece", "approval_rate", "group_approval"}
            assert set(out[m][s]["group_approval"]) == set(K.GROUPS)
    for k in ["booster_shap", "logistic_coefficients"]:
        assert set(out["stability"][k]) == {"mean_rank_correlation", "mean_top5_overlap", "n_pairs"}
