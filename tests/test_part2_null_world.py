"""Null world: base rates differ by urban/rural, nothing is biased, so the audit should stay quiet."""
import numpy as np
import pandas as pd
import pytest

from secondlook import config as C
from secondlook.audit.report import run_audit, is_quiet, null_world_check
from secondlook.contracts import FEATURES, FEATURE_DIRECTION, GROUPS
from secondlook.model.scorer import Scorer


def make_null_world(N=8000, seed=3):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"id": np.arange(N)})
    for a, l in GROUPS.items():
        df[a] = r.choice(l, N)
    df["split"] = r.choice(["train", "val", "test"], N, p=[.5, .2, .3])
    lat = r.normal(size=N) + 0.5 * (df.urban_rural == "rural")        # unequal base rates, no bias
    df["r_true"] = 1 / (1 + np.exp(-(-1.8 + 0.9 * lat)))
    df["y_true"] = (r.random(N) < df.r_true).astype(int)
    for f in FEATURES:
        df[f] = r.normal(size=N) + 0.6 * FEATURE_DIRECTION[f] * lat if f not in ("night_active_share", "local_market_share") else r.normal(size=N)
    tr, va = df.split == "train", df.split == "val"
    s = Scorer().fit(df[tr], df.y_true[tr], df[va], df.y_true[va])
    s.set_tau_cost(df[va], df.y_true[va].values)
    df["p_default"] = s.predict_proba(df)
    appr = r.random(N) < np.clip(1 - df.r_true * 3, .05, .95)
    df["approved_legacy"] = appr.astype(int)
    df["observed"] = (appr | (r.random(N) < .03)).astype(int)
    return df, s


def test_toy_null_world_is_quiet():
    df, s = make_null_world()
    out = run_audit(s.tau, bootstrap=False, df=df, scorer=s, write=False)
    # base rates differ, nothing is biased: no gap left after matching on true risk, level-down and inversion quiet
    assert not out["failure_panel"]["unexplained_gap"]
    assert not out["failure_flags"]["level_down"] and not out["failure_flags"]["inversion"]
    assert not any(p["planted_top"] for p in out["proxy_leakage"] if p["attribute"] == "gender")


def test_real_null_world_folder():
    d = C.DATA_DIR.parent / "synth_null"
    if not d.exists():
        pytest.skip("no synth_null folder")
    from secondlook.model.scorer import load_model
    quiet, out = null_world_check(d, load_model())
    assert quiet, (out["failure_flags"], out["burden_headline"])
