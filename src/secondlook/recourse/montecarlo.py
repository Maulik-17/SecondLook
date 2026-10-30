"""Chance-constrained Monte Carlo with common random numbers.

For a person, simulate K future income paths over H months, apply a candidate plan,
rescore every month with the same model and record the first month with p_default <= tau.
"""
import numpy as np
import pandas as pd

from .. import contracts as CT
from . import actions as A
from . import income_process as IP

try:  # the shared implementation wins once Part 1 provides it
    from ..model.features import cash_features
except ImportError:  # pragma: no cover - depends on branch state
    from ._features_stub import cash_features

ON_TIME = ["rent_on_time", "utility_on_time", "recharge_on_time"]
MAX_WINDOW = 24


def build_person(row_people, row_feat, hist):
    """hist: DataFrame of this person's months 1..24 (inflow, outflow, balance). Only the last w months are used."""
    w = int(round(min(max(row_feat["months_observed"], 3), MAX_WINDOW)))
    h = hist[hist.month <= MAX_WINDOW].sort_values("month").tail(w)
    w = len(h)
    return {
        "id": int(row_people["id"]),
        "occupation": row_people["occupation"],
        "w": w,
        "inc": h.inflow.values.astype(float), "out": h.outflow.values.astype(float),
        "bal": h.balance.values.astype(float),
        "feat": {f: float(row_feat[f]) for f in CT.FEATURES},
        "emi": float(row_people["loan_amount"]) / float(row_people["loan_term"]),
    }


def simulate(person, fit, prior, plan_idx, horizon, K, tau, scorer, seed, rho_override=None,
             vol_mult=1.0, on_time_effective=True):
    """Returns {"success": (K,) bool, "first": (K,) float (inf = never), "p_matrix": (K, H), "min_buffer_weeks": (K,)}."""
    rng = np.random.default_rng(seed)  # same seed for every plan: common random numbers
    Z = rng.standard_normal((K, horizon)); U = rng.random((K, horizon)); D = rng.uniform(.25, .6, (K, horizon))
    P = rng.standard_normal((K, 3))
    lv = A.plan_levels(plan_idx)
    w = person["w"]
    m, rho, sigma = IP.draw_params(fit, prior, K, P, rho_override)
    m = m - fit["m"]; m = fit["m"] + m * vol_mult  # parameter uncertainty in the mean scales with volatility
    sigma = sigma * vol_mult
    p_dip = float(np.clip(person["feat"]["dip_months"] / w, 0.01, 0.2))
    a, f = IP.fit_outflow(person["inc"], person["out"])
    floor = -0.3 * person["inc"].mean()
    emi = np.full(K, person["emi"] * (1 - lv["smaller_loan"]))

    e = np.full(K, fit["e_last"]); bal = np.full(K, person["bal"][-1])
    inc_hist = np.tile(person["inc"], (K, 1)); out_hist = np.tile(person["out"], (K, 1))
    bal_hist = np.tile(person["bal"], (K, 1))
    mean_out = person["out"].mean()
    min_buf = np.full(K, np.inf)
    frames = []
    uplift = lv["on_time"] if on_time_effective else 0.0
    for t in range(1, horizon + 1):
        e = rho * e + sigma * Z[:, t - 1]
        dip = np.where(U[:, t - 1] < p_dip, -D[:, t - 1], 0.0) * (vol_mult > 0)
        base = np.exp(m + e + dip)
        out_t = a * base * (1 - lv["cut_spend"]) + f
        inc_t = base * (1 + lv["extra_income"])
        bal = np.maximum(bal + inc_t - out_t, floor)
        min_buf = np.minimum(min_buf, bal / (mean_out / 4.345))
        inc_hist = np.column_stack([inc_hist, inc_t]); out_hist = np.column_stack([out_hist, out_t])
        bal_hist = np.column_stack([bal_hist, bal])
        n = min(w + t, MAX_WINDOW)
        cf = cash_features(inc_hist[:, -n:], out_hist[:, -n:], bal_hist[:, -n:], emi)
        d = {k: np.asarray(v, float) for k, v in cf.items()}
        for c in CT.FEATURES:
            if c not in d:
                base_v = person["feat"][c]
                d[c] = np.full(K, min(1.0, base_v + uplift) if c in ON_TIME else base_v)
        frames.append(pd.DataFrame(d)[CT.FEATURES])
    df = pd.concat(frames, ignore_index=True)
    p = np.asarray(scorer.predict_proba(df)).reshape(horizon, K).T
    approved = p <= tau
    first = np.where(approved.any(1), approved.argmax(1) + 1, np.inf)
    return {"success": first <= horizon, "first": first, "p_matrix": p, "min_buffer_weeks": min_buf}
