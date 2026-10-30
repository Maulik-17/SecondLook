"""Burden of fixing a rejection, for the creditworthy people who were rejected.

For a linear score s = a.x + b the cheapest change (quadratic effort sum_j W_j dz_j^2 in standardised units
z_j = x_j / sd_j) that lowers the score by `slack` has the closed form
    dz_j = -slack * (c_j / W_j) / sum_k (c_k^2 / W_k),   c_j = a_j * sd_j,
    effort = slack / sqrt(sum_k c_k^2 / W_k).
Each feature is then clipped to the 1st-99th percentile of the training data; if the clipped move no longer
reaches the threshold the person is infeasible (effort = inf). Effort is a diagnostic under assumed weights."""
import numpy as np
from scipy.special import logit as _logit

from ..contracts import GROUPS, FEATURE_DIRECTION
from .metrics import boot_mean, _py
from .weights import ACTIONABLE, ASSUMED, RATE_FEATURES, WEIGHTS
from .. import config as C


def threshold_for(scorer, tau):
    """Raw-logit threshold s*(tau) obtained by inverting the Platt calibration."""
    cal = scorer.cal_
    return float((_logit(tau) - cal.intercept_[0]) / cal.coef_[0, 0])


def effort_core(c, W, slack):
    """Closed-form minimum-effort move. c, W: (k,); slack: (n,). Returns effort (n,), dz (n,k)."""
    c, W, slack = np.asarray(c, float), np.asarray(W, float), np.atleast_1d(np.asarray(slack, float))
    S = np.sum(c ** 2 / W)
    effort = slack / np.sqrt(S)
    dz = -slack[:, None] * (c / W)[None, :] / S
    return effort, dz


def compute_efforts(Xa, s, a, sd, lo, hi, sstar, W):
    """Xa: (n,k) actionable raw features; s: (n,) raw logits; a: (k,) raw weights; sd, lo, hi: (k,).
    Closed form first; a feature that would leave its 1st-99th percentile range is held at the bound and the
    remaining slack is redistributed over the other features (active set). A person who still cannot reach the
    threshold with every feature at its bound gets effort = inf.
    Returns dict(effort (inf = infeasible, 0 if not rejected), slack, dz (final standardised move), rejected)."""
    slack = s - sstar
    rej = slack > 0
    n, k = Xa.shape
    eff = np.zeros(n)
    dz_all = np.zeros((n, k))
    if rej.any():
        X = Xa[rej]
        lo_r, hi_r = np.minimum(lo, X), np.maximum(hi, X)          # never push a value back inside from outside
        c = a * sd
        need = slack[rej].copy()
        free = np.ones(X.shape, bool)
        dx = np.zeros(X.shape)
        done = np.zeros(len(X), bool)
        for _ in range(k + 1):
            S = np.sum(np.where(free, c ** 2 / W, 0.0), axis=1)
            ok_S = S > 1e-15
            Sd = np.where(ok_S, S, 1.0)
            dzf = -need[:, None] * np.where(free, c / W, 0.0) / Sd[:, None]
            prop = np.where(free, dx + dzf * sd, dx)
            lower, upper = lo_r - X, hi_r - X
            viol = free & ((prop < lower - 1e-12) | (prop > upper + 1e-12))
            viol &= ~done[:, None]
            fin = ~viol.any(axis=1) & ~done
            # people with no violation: accept the proposal
            dx = np.where(fin[:, None], prop, dx)
            done |= fin
            if done.all():
                break
            # people with violations: fix violators at their bound, reduce remaining need
            act = viol.any(axis=1)
            bound = np.where(prop > upper, upper, lower)
            fixed_dx = np.where(viol, bound, 0.0)
            dx = np.where(act[:, None] & viol, fixed_dx, dx)
            need = np.where(act, need + (fixed_dx * a[None, :]).sum(axis=1), need)
            free = free & ~viol
            exhausted = act & ~free.any(axis=1)
            done |= exhausted
        s_new = s[rej] + dx @ a
        feasible = s_new <= sstar + 1e-9
        dz = dx / sd
        e = np.sqrt(np.sum(W * dz ** 2, axis=1))
        eff[rej] = np.where(feasible, e, np.inf)
        dz_all[rej] = dz
    return {"effort": eff, "slack": slack, "dz": dz_all, "rejected": rej}


class Context:
    """Training-set statistics and linear parameters used by every burden computation."""

    def __init__(self, df, scorer):
        wp = scorer.linear_params()
        feats = list(wp["features"])
        self.names = [f for f in ACTIONABLE if f in feats]
        self.feats = feats
        self.idx = [feats.index(f) for f in self.names]
        self.w_all, self.b = np.asarray(wp["w"], float), float(wp["b"])
        self.a = self.w_all[self.idx]
        tr = df[df.split == "train"] if (df.split == "train").any() else df
        T = tr[self.names]
        self.sd = T.std().values + 1e-9
        self.lo, self.hi = T.quantile(0.01).values.copy(), T.quantile(0.99).values.copy()
        for j, f in enumerate(self.names):
            if f in RATE_FEATURES:
                self.lo[j], self.hi[j] = max(self.lo[j], 0.0), min(self.hi[j], 1.0)
        self.scorer = scorer

    def W(self, wname):
        return np.array([WEIGHTS[wname][f] for f in self.names], float)

    def logit(self, df):
        return df[self.feats].values @ self.w_all + self.b


def person_efforts(df, ctx, tau, wname="base"):
    sstar = threshold_for(ctx.scorer, tau)
    out = compute_efforts(df[ctx.names].values.astype(float), ctx.logit(df), ctx.a, ctx.sd, ctx.lo, ctx.hi,
                          sstar, ctx.W(wname))
    out["sstar"] = sstar
    return out


def rg_mask(df, rejected):
    """Creditworthy (would repay) and rejected."""
    return rejected & (df.y_true.values == 0)


def burden_rows(df, eff, rg, B=200, bootstrap=True):
    """Per group level: burden = sum of finite effort over RG in g / |g| (+ bootstrap), infeasible share, n."""
    rows = []
    finite = np.where(np.isfinite(eff), eff, 0.0)
    for a, levels in GROUPS.items():
        for l in levels:
            m = (df[a] == l).values
            n = int(m.sum())
            per = finite[m] * rg[m]
            nrg = int(rg[m].sum())
            inf_n = int((rg[m] & ~np.isfinite(eff[m])).sum())
            est = float(per.mean()) if n else float("nan")
            ci = boot_mean(per, B=B) if (bootstrap and n) else [est, est, est]
            rows.append({"attribute": a, "level": _py(l), "n": n, "n_rejected_good": nrg, "burden": ci,
                         "mean_effort_rejected_good": float(finite[m][rg[m]].mean()) if nrg else None,
                         "infeasible_share": float(inf_n / nrg) if nrg else 0.0, "small_n": bool(n < C.MIN_GROUP_N)})
    return rows


def headline(rows):
    ok = [r for r in rows if r["n"] >= C.MIN_GROUP_N and np.isfinite(r["burden"][1])]
    if not ok:
        return {"worst": None, "best": None, "ratio": None}
    w = max(ok, key=lambda r: r["burden"][1])
    b = min(ok, key=lambda r: r["burden"][1])
    ratio = float(w["burden"][1] / b["burden"][1]) if b["burden"][1] > 0 else None
    return {"worst": {"attribute": w["attribute"], "level": w["level"]},
            "best": {"attribute": b["attribute"], "level": b["level"]}, "ratio": ratio}


def budget_curves(df, eff, rg, n_grid=25):
    fin = eff[rg & np.isfinite(eff)]
    q95 = float(np.percentile(fin, 95)) if len(fin) else 1.0
    grid = np.linspace(0, q95, n_grid)
    out = []
    for a, levels in GROUPS.items():
        for l in levels:
            m = (df[a] == l).values & rg
            e = eff[m]
            share = [float((e <= b).mean()) if len(e) else float("nan") for b in grid]
            out.append({"attribute": a, "level": _py(l), "n": int(m.sum()), "budget": grid.tolist(), "share": share})
    return out


def risk_matched_gap(df, eff, rejected, rows, B=200, seed=0):
    """Rejected people binned in 5 quintiles of TRUE risk r_true; within each bin the gap in mean finite effort
    between the worst-burden level of an attribute and the rest, with a bootstrap interval."""
    r = np.random.default_rng(seed)
    m = rejected & np.isfinite(eff)
    d = df[m]
    e = eff[m]
    if len(d) < 25:
        return []
    bins = np.digitize(d.r_true.values, np.quantile(d.r_true.values, [0.2, 0.4, 0.6, 0.8]))
    out = []
    for a, levels in GROUPS.items():
        cand = [x for x in rows if x["attribute"] == a and x["n"] >= C.MIN_GROUP_N]
        if not cand:
            continue
        lv = max(cand, key=lambda x: x["burden"][1])["level"]
        inl = (d[a] == lv).values
        for k in range(5):
            mk = bins == k
            x1, x0 = e[mk & inl], e[mk & ~inl]
            if len(x1) < 5 or len(x0) < 5:
                out.append({"attribute": a, "level": _py(lv), "bin": k, "n": int(len(x1) + len(x0)), "gap": [float("nan")] * 3})
                continue
            est = float(x1.mean() - x0.mean())
            bs = np.array([r.choice(x1, len(x1)).mean() - r.choice(x0, len(x0)).mean() for _ in range(B)])
            out.append({"attribute": a, "level": _py(lv), "bin": k, "n": int(len(x1) + len(x0)),
                        "gap": [float(np.percentile(bs, 2.5)), est, float(np.percentile(bs, 97.5))]})
    return out


def sensitivity(df, ctx, tau):
    """Worst group and gap sign under each weight set."""
    res = {}
    for w in WEIGHTS:
        o = person_efforts(df, ctx, tau, w)
        rg = rg_mask(df, o["rejected"])
        rows = burden_rows(df, o["effort"], rg, bootstrap=False)
        h = headline(rows)
        signs = {}
        for a, levels in GROUPS.items():
            if len(levels) == 2:
                b = {x["level"]: x["burden"][1] for x in rows if x["attribute"] == a}
                signs[a] = int(np.sign(b[_py(levels[0])] - b[_py(levels[1])]))
        res[w] = {"worst": h["worst"], "ratio": h["ratio"], "signs": signs}
    worst_ids = {str(v["worst"]) for v in res.values()}
    stable = len(worst_ids) == 1 and all(
        len({v["signs"][a] for v in res.values()}) == 1 for a in res["base"]["signs"])
    return {"by_weight_set": res, "sign_stable": bool(stable)}


def gaming_and_inversion(df, ctx, o):
    """Share of total score reduction from assumed features, and count of recommendations against expert direction."""
    rej = o["rejected"]
    if not rej.any():
        return {"gaming_ratio": 0.0, "inversion_count": 0}
    dz = o["dz"][rej]
    red = -dz * (ctx.a * ctx.sd)[None, :]            # score reduction per feature (>= 0 by construction)
    ok = np.isfinite(o["effort"][rej])
    red = red[ok]
    tot = red.sum()
    amask = np.array([f in ASSUMED for f in ctx.names])
    ratio = float(red[:, amask].sum() / tot) if tot > 0 else 0.0
    dirs = np.array([FEATURE_DIRECTION[f] for f in ctx.names])
    dx = dz[ok] * ctx.sd
    wrong = (dx * dirs[None, :]) > 1e-12              # e.g. recommends raising a feature that raises risk
    inv = int(wrong.any(axis=1).sum())
    return {"gaming_ratio": ratio, "inversion_count": inv}


def unexplained_gap(rm_rows, min_bins=3):
    """True when, for some attribute, the worst-burden level still has a higher burden than the rest after matching on
    true risk: at least `min_bins` of the 5 risk bins have a bootstrap interval entirely above zero."""
    for a in {r["attribute"] for r in rm_rows}:
        n_pos = sum(1 for r in rm_rows if r["attribute"] == a and np.isfinite(r["gap"][0]) and r["gap"][0] > 0)
        if n_pos >= min_bins:
            return True
    return False
