"""Pool vs fresh: monitoring a lender's own approved pool versus screening the whole population."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from .. import config as C
from ..audit.data import DISCLAIMER, load
from ..model.scorer import load_model


def _series(df, scorer, eps, rounds, batch, seed):
    r = np.random.default_rng(seed)
    feats, tau = list(scorer.features), scorer.tau
    mu, sd = df[feats].mean().values, df[feats].std().values + 1e-9
    z = lambda d: (d[feats].values - mu) / sd
    fresh = df[df.split == "test"]
    Zf, yf = z(fresh), fresh.y_true.values
    stream = df[df.split == "train"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    batch = int(max(20, min(batch, len(stream) // (rounds + 1))))
    pool_Z, pool_y = [], []

    def fit():
        y = np.concatenate(pool_y)
        if len(np.unique(y)) < 2:
            return None
        return LogisticRegression(max_iter=500).fit(np.vstack(pool_Z), y)

    def recall(m, Z, y):
        if m is None or (y == 1).sum() == 0:
            return float("nan")
        return float((m.predict_proba(Z[y == 1])[:, 1] > tau).mean())

    first = stream.iloc[:batch]
    obs0 = first[first.approved_legacy == 1]
    pool_Z.append(z(obs0)); pool_y.append(obs0.y_true.values)
    model = fit()
    res = {"round": [0], "pool_recall": [recall(model, np.vstack(pool_Z), np.concatenate(pool_y))],
           "fresh_recall": [recall(model, Zf, yf)]}
    for k in range(1, rounds + 1):
        b = stream.iloc[k * batch:(k + 1) * batch]
        if len(b) == 0:
            break
        Zb = z(b)
        appr = (model.predict_proba(Zb)[:, 1] <= tau) if model is not None else np.ones(len(b), bool)
        expl = (~appr) & (r.random(len(b)) < eps)
        keep = appr | expl
        pool_Z.append(Zb[keep]); pool_y.append(b.y_true.values[keep])
        model = fit() or model
        res["round"].append(k)
        res["pool_recall"].append(recall(model, np.vstack(pool_Z), np.concatenate(pool_y)))
        res["fresh_recall"].append(recall(model, Zf, yf))
    return res


def run_poolfresh(eps, rounds=20, batch=1500, seed=0, df=None, scorer=None, write=True):
    df = load() if df is None else df
    scorer = load_model() if scorer is None else scorer
    series = {}
    for e in sorted({0.0, 0.03, 0.10, float(eps)}):
        series[f"{e:g}"] = _series(df, scorer, e, rounds, batch, seed)
    out = dict(series[f"{float(eps):g}"])
    out.update({"eps": float(eps), "series": series, "tau": float(scorer.tau), "disclaimer": DISCLAIMER})
    if write:
        C.ART_DIR.mkdir(parents=True, exist_ok=True)
        with open(C.ART_DIR / "poolfresh.json", "w") as fh:
            json.dump(out, fh, default=float)
    return out
