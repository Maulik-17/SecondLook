"""Berka sanity run: sign-constrained scorecard, 5-fold CV AUC with bootstrap CI, approval rates by gender and district tercile,
and calibration targets. Writes ART_DIR/berka_sanity.json and ART_DIR/berka_calibration.json. Uses a stdlib parser (no database)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from secondlook import config, contracts as K
from secondlook.berka.features import CASH_COLS, build
from secondlook.berka.load import DEFAULT_PATH, table

CACHE = Path("/tmp/berka_cache")


def get(name, path):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = table(path, name)
    d.to_parquet(f, index=False)
    return d


def fit_constrained(X, y):
    feats = list(CASH_COLS)
    while True:
        mu, sd = X[feats].mean(), X[feats].std() + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=2000).fit((X[feats] - mu) / sd, y)
        bad = [f for f, w in zip(feats, lr.coef_[0]) if K.FEATURE_DIRECTION[f] * w < 0]
        if not bad:
            return feats, mu, sd, lr
        feats.remove(max(bad, key=lambda f: abs(dict(zip(feats, lr.coef_[0]))[f])))


def main(path=DEFAULT_PATH):
    loan, trans = get("loan", path), get("trans", path)
    X = build(loan, trans).reset_index(drop=True)
    assert (X["date_used_max"] < X["loan_date"]).all()
    y = X["y"].values
    oof = np.zeros(len(X))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=config.SEED % 2**31).split(X, y):
        f, mu, sd, lr = fit_constrained(X.iloc[tr], y[tr])
        oof[te] = lr.predict_proba((X.iloc[te][f] - mu) / sd)[:, 1]
    r = np.random.default_rng(config.SEED)
    boots = []
    for _ in range(1000):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], oof[i]))
    auc = [float(np.quantile(boots, .025)), float(roc_auc_score(y, oof)), float(np.quantile(boots, .975))]

    disp, client, acc, dist = get("disp", path), get("client", path), get("account", path), get("district", path)
    owner = disp[disp.type == "OWNER"].merge(client[["client_id", "gender"]], on="client_id")[["account_id", "gender"]]
    meta = X.merge(owner, on="account_id", how="left").merge(acc[["account_id", "district_id"]], on="account_id", how="left") \
        .merge(dist[["district_id", "A11"]], on="district_id", how="left")
    meta["p"] = oof
    meta["approve"] = meta.p <= np.quantile(oof, 0.8)               # approve the lowest-risk 80% (illustration only)
    meta["tercile"] = pd.qcut(meta.A11, 3, labels=["low", "mid", "high"])
    by = lambda g: [{"level": str(k), "n": int(len(v)), "approval": float(v.approve.mean()), "default_rate": float(v.y.mean())}
                    for k, v in meta.groupby(g, observed=True)]
    out = {"n": int(len(X)), "defaults": int(y.sum()), "auc": auc, "window_months": 6, "cv": "5-fold, out-of-fold predictions",
           "features": CASH_COLS, "approval_rule": "lowest-risk 80% approved (illustration)",
           "by_gender": by("gender"), "by_district_income_tercile": by("tercile"),
           "note": "With 76 defaults the interval is wide. Berka is Czech bank data from the 1990s, not Indian data."}
    config.ART_DIR.mkdir(parents=True, exist_ok=True)
    (config.ART_DIR / "berka_sanity.json").write_text(json.dumps(out, indent=2))

    # calibration targets vs simulator
    from secondlook.sim.generate import generate
    S = generate(4000, 1)["features"]
    qs = [.1, .25, .5, .75, .9]
    cal = {c: {"berka": [float(x) for x in X[c].quantile(qs)], "sim": [float(x) for x in S[c].quantile(qs)]}
           for c in ["inflow_cv", "dip_months", "weeks_of_buffer", "expense_to_income", "emi_to_income"]}
    (config.ART_DIR / "berka_calibration.json").write_text(json.dumps({"quantiles": qs, "targets": cal}, indent=2))
    print(json.dumps({k: out[k] for k in ["n", "defaults", "auc", "by_gender"]}, indent=1))
    for c, v in cal.items():
        print(c, "berka", np.round(v["berka"], 2), "sim", np.round(v["sim"], 2))


if __name__ == "__main__":
    main()
