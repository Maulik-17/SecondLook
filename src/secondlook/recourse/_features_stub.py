"""Local stand-in for secondlook.model.features.cash_features with the same signature.
Used only until the shared implementation is importable (see montecarlo.py); kept in step with it
so that scores computed on simulated paths use the same feature definitions as the scorer's training data."""
import numpy as np


def cash_features(inc, out, bal, emi):
    inc, out, bal = (np.asarray(a, float) for a in (inc, out, bal))
    K, w = inc.shape
    mi, mo = inc.mean(1), out.mean(1)
    med = np.median(inc, axis=1)
    tt = np.arange(w) - (w - 1) / 2
    slope = ((bal - bal.mean(1, keepdims=True)) * tt).sum(1) / max((tt ** 2).sum(), 1e-9) / mi
    return {
        "mean_inflow": mi, "median_inflow": med, "inflow_cv": inc.std(1) / mi,
        "dip_months": (inc < 0.75 * med[:, None]).sum(1).astype(float),
        "spike_months": (inc > 1.25 * med[:, None]).sum(1).astype(float),
        "weeks_of_buffer": bal.mean(1) / (mo / 4.345),
        "low_balance_days": 30 * (bal < 0.25 * mo[:, None]).mean(1),
        "expense_to_income": out.sum(1) / inc.sum(1),
        "balance_level": bal.mean(1) / mi, "balance_slope": slope,
        "months_observed": np.full(K, float(w)), "emi_to_income": np.asarray(emi, float) / mi,
    }
