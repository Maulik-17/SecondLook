"""Weighted Mondrian split conformal, written by hand (exploration weights are not supported by off-the-shelf tools).
Nonconformity of label y is 1 - p_hat(y | x): label 0 scores p_default, label 1 scores 1 - p_default."""
import json

import joblib
import numpy as np

from .. import config as C
from ..contracts import GROUPS
from ..audit.data import DISCLAIMER, load, test_pop
from .exploration import redraw

CELLS = ["thin_file", "gender"]          # Mondrian partition


def wquantile(s, w, level, w_test):
    """Smallest s_k such that weight of {s <= s_k} / (sum w + w_test) >= level; inf if never."""
    o = np.argsort(s, kind="stable")
    s, w = np.asarray(s)[o], np.asarray(w)[o]
    cum = np.cumsum(w) / (w.sum() + w_test)
    k = np.searchsorted(cum, level - 1e-12)
    return float(s[k]) if k < len(s) else np.inf


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


class ConformalModel:
    def fit(self, df_cal, weights, alpha, cells=CELLS):
        """df_cal: observed calibration rows (columns p_default, y_obs, + cell columns). weights: array, capped at W_MAX."""
        weights = np.minimum(np.asarray(weights, float), C.W_MAX)
        self.alpha, self.cells, self.q, self.ess, self.n_cell = alpha, list(cells), {}, {}, {}
        s = np.where(df_cal.y_obs.values == 1, 1 - df_cal.p_default.values, df_cal.p_default.values)
        for key, idx in df_cal.reset_index(drop=True).groupby(self.cells).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            w = weights[idx]
            self.q[key] = wquantile(s[idx], w, 1 - alpha, w.mean())
            self.ess[key] = ess(w)
            self.n_cell[key] = int(len(idx))
        self.q_global = wquantile(s, weights, 1 - alpha, weights.mean())
        return self

    def predict_set(self, p_default, groups):
        out = []
        p_default = np.atleast_1d(p_default)
        keys = list(zip(*[groups[c].values for c in self.cells]))
        for p, k in zip(p_default, keys):
            q = self.q.get(tuple(k), self.q_global)
            sets = [y for y, sc in ((0, p), (1, 1 - p)) if sc <= q]
            out.append({"set": sets, "abstain": len(sets) != 1})
        return out


def load_conformal() -> ConformalModel:
    return joblib.load(C.ART_DIR / "conformal.joblib")


def _variant(name, cal, obs, explored, prop, te, alpha):
    """Fit one calibration variant and evaluate it on the fully-labelled test population."""
    if name == "unweighted_accepts":
        keep = obs & (cal.approved_legacy.values == 1)
        w = np.ones(keep.sum())
    elif name == "unweighted_explored":
        keep = explored
        w = np.ones(keep.sum())
    else:
        keep = obs
        w = np.minimum(1 / prop[keep], C.W_MAX)
    d = cal[keep].copy()
    d["y_obs"] = d.y_true.values
    if len(d) < 50:
        return None, {"status": "too few rows", "n_cal": int(len(d)), "groups": [], "coverage_gap": None, "referral_gap": None}
    m = ConformalModel().fit(d, w, alpha)
    res = m.predict_set(te.p_default.values, te)
    abst = np.array([r["abstain"] for r in res])
    cover = np.array([int(y) in r["set"] for y, r in zip(te.y_true.values, res)])
    rows = []
    dpos = d.reset_index(drop=True)
    for a, levels in GROUPS.items():
        for l in levels:
            mk = (te[a] == l).values
            ml = (dpos[a] == l).values
            rows.append({"attribute": a, "level": l if not hasattr(l, "item") else l.item(), "n": int(mk.sum()),
                         "coverage": float(cover[mk].mean()) if mk.any() else None,
                         "referral": float(abst[mk].mean()) if mk.any() else None,
                         "ess": ess(w[ml]) if ml.any() else 0.0, "n_cal": int(ml.sum())})
    ok = [r for r in rows if r["n"] >= C.MIN_GROUP_N]
    gap = lambda k: max([max(x[k] for x in ok if x["attribute"] == a) - min(x[k] for x in ok if x["attribute"] == a)
                         for a in GROUPS if any(x["attribute"] == a for x in ok)] or [0.0])
    return m, {"status": "ok", "n_cal": int(len(d)), "overall_coverage": float(cover.mean()),
               "overall_referral": float(abst.mean()), "groups": rows,
               "coverage_gap": float(gap("coverage")), "referral_gap": float(gap("referral")),
               "cell_ess": {"|".join(map(str, k)): v for k, v in m.ess.items()}}


def run_uncertainty(eps, alpha, df=None, write=True, seed=0):
    df = load() if df is None else df
    cal = df[df.split == "val"].reset_index(drop=True)
    te = test_pop(df)
    explored, obs, prop = redraw(cal, eps, seed=seed)
    variants, models = {}, {}
    for name in ("unweighted_accepts", "unweighted_explored", "weighted"):
        models[name], variants[name] = _variant(name, cal, obs, explored, prop, te, alpha)
    warns = []
    if eps <= 0.01:
        warns.append(f"eps = {eps}: very few explored rejects; weights are large and the weighted sets are unstable.")
    for name, v in variants.items():
        if v.get("status") != "ok":
            warns.append(f"{name}: {v['status']} (n_cal={v['n_cal']}).")
            continue
        low = [k for k, e in v["cell_ess"].items() if e < 30]
        if low:
            warns.append(f"{name}: effective sample size below 30 in cell(s) {', '.join(low)}.")
    try:
        from ..model.scorer import load_model
        ch = str(getattr(load_model(), "config_hash", "unset"))
    except Exception:
        ch = "unset"
    out = {"eps": float(eps), "alpha": float(alpha), "config_hash": ch, "variants": variants, "warnings": warns,
           "w_max": C.W_MAX, "disclaimer": DISCLAIMER}
    if variants["weighted"].get("status") == "ok":
        from .gamma import gamma_sensitivity
        d = cal[obs].copy()
        d["y_obs"] = d.y_true.values
        out["gamma_sensitivity"] = gamma_sensitivity(d, np.minimum(1 / prop[obs], C.W_MAX), te, alpha)
    if write:
        C.ART_DIR.mkdir(parents=True, exist_ok=True)
        with open(C.ART_DIR / "uncertainty.json", "w") as fh:
            json.dump(out, fh, default=float)
        if models["weighted"] is not None and abs(eps - C.EXPLORATION_EPS) < 1e-9:
            joblib.dump(models["weighted"], C.ART_DIR / "conformal.joblib")
    return out
