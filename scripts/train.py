"""Fits the sign-constrained scorecard, calibrates, sets the cost-based tau, writes scores, model and metrics."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from secondlook import config, contracts as K
from secondlook.model.scorer import Scorer, save_model

DISCLAIMER = "Synthetic data; an audit-tool demonstration, not a lending decision tool."
DESC = {
    "mean_inflow": "Average money received per month over the observed window",
    "median_inflow": "Median money received per month over the observed window",
    "inflow_cv": "Month-to-month variation in income (standard deviation divided by mean)",
    "dip_months": "Months where income fell below 75% of the median month",
    "spike_months": "Months where income rose above 125% of the median month",
    "weeks_of_buffer": "Average month-end balance divided by average weekly spending",
    "low_balance_days": "Days per month the balance was under a quarter of monthly spending",
    "expense_to_income": "Total spending divided by total income",
    "balance_level": "Average balance divided by average monthly income",
    "balance_slope": "Trend of the balance per month, relative to monthly income",
    "rent_on_time": "Share of rent payments made on time (assumed signal, not shown to predict default)",
    "utility_on_time": "Share of utility bills paid on time (assumed signal, not shown to predict default)",
    "recharge_on_time": "Share of phone recharges made on time (assumed signal, not shown to predict default)",
    "months_observed": "Number of months of account history available",
    "emi_to_income": "Monthly instalment divided by average monthly income",
    "night_active_share": "Share of activity at night (planted proxy for gender; no effect on true risk)",
    "local_market_share": "Share of activity in the local market (planted proxy for region; no effect on true risk)",
}


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def auc_ci(y, p, B=500, seed=config.SEED):
    r = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(B):
        i = r.integers(0, n, n)
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return [float(np.quantile(vals, .025)), float(roc_auc_score(y, p)), float(np.quantile(vals, .975))]


def main():
    D, A = config.DATA_DIR, config.ART_DIR
    A.mkdir(parents=True, exist_ok=True)
    ppl, F = pd.read_parquet(D / "people.parquet"), pd.read_parquet(D / "features.parquet")
    df = ppl.merge(F, on="id")
    sp = {s: df[df.split == s] for s in ["train", "val", "test", "oot"]}
    y = {s: sp[s].y_true.values for s in sp}

    feats, dropped = list(K.FEATURES), []
    while True:
        s = Scorer(feats).fit(sp["train"], y["train"], sp["val"], y["val"])
        w = dict(zip(s.features, s.lr_.coef_[0]))
        bad = [f for f in feats if K.FEATURE_DIRECTION[f] * w[f] < 0]
        if not bad:
            break
        worst = max(bad, key=lambda f: abs(w[f]))     # drop one at a time: removing the largest violator often fixes the rest
        dropped.append(worst)
        feats = [f for f in feats if f != worst]
    if config.TARGET_APPROVAL_RATE is not None:
        s.tau = float(np.quantile(s.predict_proba(sp["val"]), config.TARGET_APPROVAL_RATE))
    else:
        s.set_tau_cost(sp["val"], y["val"])

    sc = pd.DataFrame({"id": df.id.values, "p_default": s.predict_proba(df), "logit": s.logit(df), "tau": s.tau})
    sc["decision"] = (sc.p_default <= s.tau).astype(int)
    sc = sc.sort_values("id").reset_index(drop=True)
    K.check("scores", sc).to_parquet(D / "scores.parquet", index=False)

    s.config_hash = hashlib.sha1(json.dumps({k: str(v) for k, v in vars(config).items() if k.isupper()}, sort_keys=True).encode()).hexdigest()[:10]
    s.dropped_features = dropped
    save_model(s)

    m = {"config_hash": s.config_hash, "disclaimer": DISCLAIMER, "kept_features": s.features, "dropped_features": dropped,
         "tau": s.tau, "approval_rate_val": float(s.decide(sp["val"]).mean())}
    for name in ["test", "oot"]:
        p = s.predict_proba(sp[name])
        m[name] = {"n": int(len(p)), "auc": auc_ci(y[name], p), "brier": float(brier_score_loss(y[name], p)),
                   "ece": ece(p, y[name]), "base_rate": float(y[name].mean())}
    pv = s.predict_proba(sp["val"])
    m["val"] = {"n": int(len(pv)), "auc": float(roc_auc_score(y["val"], pv)), "ece": ece(pv, y["val"])}
    pt = s.predict_proba(sp["test"])
    m["calibration_by_gender_test"] = {g: {"n": int(msk.sum()), "mean_pred": float(pt[msk].mean()), "observed": float(y["test"][msk].mean()),
                                           "ece": ece(pt[msk], y["test"][msk])}
                                       for g in ["F", "M"] for msk in [(sp["test"].gender == g).values]}
    (A / "metrics.json").write_text(json.dumps(m, indent=2))
    (A / "oot.json").write_text(json.dumps({"n": m["oot"]["n"], "auc": m["oot"]["auc"], "test_auc": m["test"]["auc"]}, indent=2))
    dd = [{"feature": f, "source": K.FEATURE_SOURCE[f], "description": DESC[f], "direction": K.FEATURE_DIRECTION[f]} for f in K.FEATURES]
    (A / "data_dictionary.json").write_text(json.dumps(dd, indent=2))
    print(json.dumps({k: m[k] for k in ["tau", "test", "oot", "dropped_features"]}, indent=1))


if __name__ == "__main__":
    main()
