"""AR(1) log-income model with shrinkage for short series."""
import joblib
import numpy as np
import pandas as pd

from .. import config as C


def fit_ar1(x):
    """Closed-form fit on a short log-income series. Returns dict(m, rho, sigma, e_last, se_rho, w)."""
    x = np.asarray(x, float)
    w = len(x); m = x.mean(); d = x - m
    rho = float(np.clip((d[1:] * d[:-1]).sum() / max((d[:-1] ** 2).sum(), 1e-9), 0.0, 0.9)) if w > 2 else 0.0
    resid = d[1:] - rho * d[:-1] if w > 2 else d
    sigma = float(max(resid.std(ddof=1) if len(resid) > 1 else 0.2, 0.03))
    return {"m": float(m), "rho": rho, "sigma": sigma, "e_last": float(d[-1]),
            "se_rho": (1 - rho ** 2) / np.sqrt(max(w, 2)), "w": w}


def _fit_matrix(X):
    """Vectorised fit_ar1 over rows of X (n, w)."""
    d = X - X.mean(1, keepdims=True)
    rho = np.clip((d[:, 1:] * d[:, :-1]).sum(1) / np.maximum((d[:, :-1] ** 2).sum(1), 1e-9), 0, 0.9)
    resid = d[:, 1:] - rho[:, None] * d[:, :-1]
    sigma = np.maximum(resid.std(1, ddof=1), 0.03)
    return rho, sigma


def population_prior(monthly, people, max_people=None, seed=C.SEED):
    """Fit people on all 24 history months; mean and sd of (rho, log sigma) per occupation."""
    ids = people["id"].values
    if max_people and len(ids) > max_people:
        ids = np.random.default_rng(seed).choice(ids, max_people, replace=False)
    h = monthly[(monthly.month <= C.HISTORY_MONTHS) & monthly.id.isin(ids)]
    piv = h.pivot(index="id", columns="month", values="inflow").dropna()
    rho, sig = _fit_matrix(np.log(np.maximum(piv.values, 1.0)))
    d = pd.DataFrame({"id": piv.index, "rho": rho, "logsig": np.log(sig)}).merge(people[["id", "occupation"]], on="id")
    out = d.groupby("occupation").agg(rho_mean=("rho", "mean"), rho_sd=("rho", "std"),
                                      ls_mean=("logsig", "mean"), ls_sd=("logsig", "std")).fillna(0.1)
    return out.to_dict("index")


def load_prior(monthly=None, people=None):
    """Cached in ART_DIR/income_prior.joblib; computed once if missing."""
    path = C.ART_DIR / "income_prior.joblib"
    if path.exists():
        return joblib.load(path)
    prior = population_prior(monthly, people, max_people=5000)
    try:
        C.ART_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(prior, path)
    except OSError:
        pass
    return prior


def shrink(fit, pri):
    lam = fit["w"] / (fit["w"] + 8)
    rho_post = lam * fit["rho"] + (1 - lam) * pri["rho_mean"]
    ls_post = lam * np.log(fit["sigma"]) + (1 - lam) * pri["ls_mean"]
    return lam, float(rho_post), float(ls_post)


def draw_params(fit, pri, K, P, rho_override=None):
    """K parameter draws from pre-drawn standard normals P (K, 3), so they are common across plans."""
    w = fit["w"]
    lam, rho_post, ls_post = shrink(fit, pri)
    if rho_override is not None:
        rho = np.full(K, float(rho_override))
    else:
        rho = np.clip(rho_post + (fit["se_rho"] * (1 - lam) + 0.05) * P[:, 0], 0, 0.95)
    sigma = np.exp(ls_post + P[:, 1] / np.sqrt(2 * max(w - 1, 1)))
    m = fit["m"] + sigma / np.sqrt(w) * P[:, 2]
    return m, rho, sigma


def fit_outflow(inc, out):
    """out = a*inc + f by OLS; a clipped to [0.3, 0.95], f >= 0."""
    inc, out = np.asarray(inc, float), np.asarray(out, float)
    if len(inc) < 3 or inc.std() < 1e-9:
        a = float(np.clip(out.sum() / max(inc.sum(), 1e-9), 0.3, 0.95))
    else:
        a = float(np.clip(np.cov(inc, out, ddof=0)[0, 1] / inc.var(), 0.3, 0.95))
    f = float(max(out.mean() - a * inc.mean(), 0.0))
    return a, f
