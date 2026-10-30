import numpy as np

from .. import config as C
from ..contracts import GROUPS


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(p), float(c + h)]


def boot(stat, df, B=1000, seed=0):
    """Percentile bootstrap of stat(df) -> float. Returns [lo, est, hi]."""
    r = np.random.default_rng(seed)
    n = len(df)
    vals = np.array([stat(df.iloc[r.integers(0, n, n)]) for _ in range(B)])
    return [float(np.nanpercentile(vals, 2.5)), float(stat(df)), float(np.nanpercentile(vals, 97.5))]


def boot_mean(x, B=1000, seed=0):
    """Percentile bootstrap of the mean of a 1-d array. Returns [lo, est, hi]."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return [float("nan")] * 3
    r = np.random.default_rng(seed)
    vals = x[r.integers(0, n, (B, n))].mean(axis=1)
    return [float(np.percentile(vals, 2.5)), float(x.mean()), float(np.percentile(vals, 97.5))]


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def _py(l):
    return l.item() if hasattr(l, "item") else l


def group_table(df, tau, B=200, bootstrap=True):
    """One row per (attribute, level). Positive = approved.
    ppv = share of approved who repay; tpr = P(approved | repays); fpr = P(approved | defaults).
    Outcome metrics are 'among approved' / use the generator's true outcome; missing_outcome_share is the
    share of the group the lender never observed an outcome for."""
    rows = []
    for a, levels in GROUPS.items():
        for l in levels:
            sub = df[df[a] == l]
            n = len(sub)
            appr = (sub.p_default <= tau).values
            good = (sub.y_true == 0).values
            bad = ~good
            rows.append({
                "attribute": a, "level": _py(l), "n": int(n), "n_approved": int(appr.sum()),
                "approval": wilson(int(appr.sum()), n),
                "base_rate": wilson(int(bad.sum()), n),
                "ppv": wilson(int((appr & good).sum()), int(appr.sum())),
                "tpr": wilson(int((appr & good).sum()), int(good.sum())),
                "fpr": wilson(int((appr & bad).sum()), int(bad.sum())),
                "ece": ece(sub.p_default.values, sub.y_true.values) if n else float("nan"),
                "brier": float(((sub.p_default - sub.y_true) ** 2).mean()) if n else float("nan"),
                "missing_outcome_share": float(1 - sub.observed.mean()) if n else float("nan"),
                "small_n": bool(n < C.MIN_GROUP_N),
            })
    return rows


def disparate_impact(rows):
    """min/max of approval rates over levels with n >= MIN_GROUP_N, one attribute at a time."""
    out = {}
    for a in GROUPS:
        r = [x["approval"][1] for x in rows if x["attribute"] == a and x["n"] >= C.MIN_GROUP_N]
        out[a] = float(min(r) / max(r)) if r and max(r) > 0 else None
    return out
