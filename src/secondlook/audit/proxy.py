"""Proxy-leakage card: can a group attribute be reconstructed from the model's features?"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

PLANTED = {"gender": "night_active_share", "region": "local_market_share"}
ATTRS = ["gender", "urban_rural", "region"]


def _auc(y, P):
    if P.shape[1] == 2:
        return roc_auc_score(y, P[:, 1])
    return roc_auc_score(y, P, multi_class="ovr", average="macro")


def _boot_auc(y, P, B, r):
    n = len(y)
    v = []
    for _ in range(B):
        i = r.integers(0, n, n)
        try:
            v.append(_auc(y[i], P[i]))
        except ValueError:
            pass
    return [float(np.percentile(v, 2.5)), float(_auc(y, P)), float(np.percentile(v, 97.5))]


def proxy_leakage(df, scorer, tau, B=200, n_perm=100, seed=0, fast=False):
    """df: rows with features, groups, r_true, p_default (test + val rows are used)."""
    r = np.random.default_rng(seed)
    d = df[df.split.isin(["test", "val"])].reset_index(drop=True)
    feats = list(scorer.features)
    X = d[feats].values.astype(float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = []
    for A in ATTRS:
        classes, y = np.unique(d[A].values, return_inverse=True)
        lr = LogisticRegression(max_iter=1000)
        P1 = cross_val_predict(lr, Xz, y, cv=5, method="predict_proba")
        P2 = cross_val_predict(HistGradientBoostingClassifier(max_iter=30 if fast else 100, random_state=seed), Xz, y, cv=3 if fast else 5, method="predict_proba")
        auc, auc_b = _boot_auc(y, P1, B, r), _boot_auc(y, P2, B, r)
        sub = r.choice(len(d), min(5000, len(d)), replace=False)
        null = []
        for _ in range(n_perm):
            ys = r.permutation(y[sub])
            Pn = cross_val_predict(LogisticRegression(max_iter=300), Xz[sub], ys, cv=3, method="predict_proba")
            null.append(_auc(ys, Pn))
        null = np.array(null)
        p_val = float((np.sum(null >= auc[1]) + 1) / (len(null) + 1))
        planted_top = False
        if A in PLANTED and PLANTED[A] in feats:
            att = LogisticRegression(max_iter=1000).fit(Xz, y)
            imp = permutation_importance(att, Xz[sub], y[sub], n_repeats=3, random_state=seed, scoring="roc_auc_ovr" if len(classes) > 2 else "roc_auc")
            planted_top = bool(feats[int(np.argmax(imp.importances_mean))] == PLANTED[A])
        out.append({"attribute": A, "n": int(len(d)), "auc": auc, "auc_boosted": auc_b,
                    "permutation_null_95": float(np.percentile(null, 95)), "p_value": p_val,
                    "planted_top": planted_top, "conditional_gap": conditional_gap(d, A, tau, r)})
    return out


def conditional_gap(d, A, tau, r, B=100):
    """Equal-risk check: bin by true-risk deciles, approval gap (max - min over levels) per bin, mean absolute gap."""
    t = d[d.split == "test"] if (d.split == "test").any() else d
    bins = np.digitize(t.r_true.values, np.quantile(t.r_true.values, np.linspace(0.1, 0.9, 9)))
    ap = (t.p_default.values <= tau).astype(float)
    g = t[A].values
    levels = np.unique(g)

    def stat(ix):
        gaps = []
        for k in range(10):
            m = ix[bins[ix] == k]
            rates = [ap[m][g[m] == l].mean() for l in levels if (g[m] == l).sum() >= 5]
            if len(rates) >= 2:
                gaps.append(max(rates) - min(rates))
        return float(np.mean(gaps)) if gaps else float("nan")

    idx = np.arange(len(t))
    v = [stat(r.integers(0, len(t), len(t))) for _ in range(B)]
    return [float(np.nanpercentile(v, 2.5)), stat(idx), float(np.nanpercentile(v, 97.5))]
