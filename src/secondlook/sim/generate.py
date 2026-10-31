"""Synthetic gig-worker simulator. Hidden truth: default depends on the simulated future shortfall rate
(unobserved post-loan shocks), not on any group label. Bias switches: config.BIAS_FLAGS or the `bias` argument."""
import numpy as np
import pandas as pd

from .. import config as C
from ..contracts import FEATURES
from ..model.features import cash_features

OCC = ["gig_driver", "gig_delivery", "shopkeeper", "student_parttime", "salaried_informal"]
P_OCC = np.array([.25, .20, .25, .10, .20])
MU = np.log([18000, 16000, 25000, 8000, 20000])      # monthly income level by occupation (assumption)
SIG = np.array([.30, .35, .28, .40, .15])            # income volatility by occupation (assumption)
RHO_A = np.array([2, 2, 3, 2, 4.]); RHO_B = np.array([3, 3, 3, 3, 3.])   # persistence ~ Beta(a, b)
SHOCK = np.array([.06, .07, .05, .08, .02])          # monthly dip probability by occupation
P_FEMALE = np.array([.12, .15, .40, .55, .45])


def generate(n, seed, cohort="main", T=C.HISTORY_MONTHS, H=C.OUTCOME_MONTHS, K=20, eps=C.EXPLORATION_EPS, bias=None):
    """cohort: 'main' | 'oot' (drifted volatility and shocks) | 'null' (no bias, unequal base rates).
    bias: None = use config.BIAS_FLAGS. Returns dict of DataFrames: people, features, monthly, latents, lender_history."""
    r = np.random.default_rng(seed)
    if bias is None:
        bias = C.BIAS_FLAGS
    bias = bool(bias) and cohort != "null"
    sigma_scale = 1.2 if cohort == "oot" else 1.0
    shock_scale = 1.3 if cohort == "oot" else 1.0

    occ = r.choice(5, n, p=P_OCC)
    fem = r.random(n) < P_FEMALE[occ]
    rural = r.random(n) < .45
    region = np.where(rural, r.choice(6, n, p=[.05, .1, .15, .2, .25, .25]), r.choice(6, n, p=[.3, .25, .2, .12, .08, .05]))
    shared = r.random(n) < np.where(rural & fem, .4, .2)
    docp = r.random(n) < .3
    thin = r.random(n) < np.where((occ == 3) | (occ < 2), .6, .4)
    mu = np.exp(MU[occ] + .25 * r.standard_normal(n))
    sigma = SIG[occ] * np.exp(.25 * r.standard_normal(n)) * sigma_scale
    if bias:                                          # injected bias 2: women in gig work have higher volatility
        sigma = np.where(fem & (occ < 2), sigma * 1.25, sigma)
    rho = r.beta(RHO_A[occ], RHO_B[occ])
    shock_p = SHOCK[occ] * shock_scale
    alpha = r.uniform(.55, .78, n); fixed = mu * r.uniform(.08, .18, n)
    amp = r.uniform(.05, .2, n); ph = r.uniform(0, 2 * np.pi, n)
    emi = mu * .22 * C.LTI_SCALE * np.exp(.35 * r.standard_normal(n))
    bal0 = mu * r.uniform(.1, 1.5, n)

    def run(Tn, e, bal, start, emi_pay=None):
        inc = np.zeros((Tn, n)); out = np.zeros((Tn, n)); b = np.zeros((Tn, n)); sf = np.zeros((Tn, n))
        dip = np.zeros(n); dipleft = np.zeros(n)
        for t in range(Tn):
            e = rho * e + np.sqrt(1 - rho ** 2) * sigma * r.standard_normal(n)       # AR(1) income residual
            new = (dipleft <= 0) & (r.random(n) < shock_p)
            dipleft = np.where(new, r.integers(1, 4, n), dipleft)
            dip = np.where(new, -r.uniform(.25, .6, n), dip)
            sh = np.where(dipleft > 0, dip, 0.); dipleft = dipleft - 1
            i = np.exp(np.log(mu) + amp * np.sin(2 * np.pi * (start + t) / 12 + ph) + e + sh)
            o = alpha * i + fixed * (1 + .1 * r.standard_normal(n))
            if emi_pay is not None:                                                  # unobserved shock after the loan
                o = o + (r.random(n) < C.POST_SHOCK_P) * mu * r.uniform(.3, C.POST_SHOCK_MAX, n)
            avail = bal + i - o
            if emi_pay is not None:
                paid = np.minimum(emi_pay, np.maximum(avail, 0)); sf[t] = emi_pay - paid; avail = avail - paid
            bal = np.maximum(avail, -.3 * mu)
            inc[t], out[t], b[t] = i, o, bal
        return inc, out, b, sf, e, bal

    e0 = sigma * r.standard_normal(n)
    inc, out, bal, _, e, balT = run(T, e0, bal0, 0)
    dflt = np.zeros((K, n), bool); keep = None
    for k in range(K):                                  # K simulated futures per person
        i2, o2, b2, sf, _, _ = run(H, e.copy(), balT.copy(), T, emi_pay=emi)
        dflt[k] = ((sf > 1e-9).sum(0) > 2) | (sf.sum(0) > 2.0 * emi)   # >2 months short, or total shortfall > 2 EMI
        if k == 0:
            keep = (i2, o2, b2)
    r_mc = np.clip(dflt.mean(0), .02, .98)
    lg = C.LOGIT_SLOPE * np.log(r_mc / (1 - r_mc)) + C.LOGIT_INTERCEPT + C.LOGIT_FRAILTY_SD * r.standard_normal(n)
    if cohort == "null":                                # unequal base rates, still no discrimination by the scorer
        lg = lg + .35 * rural
    r_true = 1 / (1 + np.exp(-lg))
    y = (r.random(n) < r_true).astype(int)

    # ---- observed features (thin-file people see only 3 to 5 months)
    m = np.where(thin, r.integers(3, 6, n), T)
    cols = ["mean_inflow", "median_inflow", "inflow_cv", "dip_months", "spike_months", "weeks_of_buffer", "low_balance_days",
            "expense_to_income", "balance_level", "balance_slope", "months_observed", "emi_to_income"]
    F = {c: np.zeros(n) for c in cols + ["rent_on_time", "utility_on_time", "recharge_on_time"]}
    z_risk = (lg - lg.mean()) / lg.std()               # ASSUMPTION: payment discipline reflects the hidden risk, which cash features do not capture
    for mm in np.unique(m):
        idx = np.where(m == mm)[0]
        I, O, B = inc[T - mm:, idx].T, out[T - mm:, idx].T, bal[T - mm:, idx].T
        for c, v in cash_features(I, O, B, emi[idx]).items():
            F[c][idx] = v
        cash = (B / O.mean(1, keepdims=True) - 1).mean(1)
        for nm, base in [("rent_on_time", .6), ("utility_on_time", .65), ("recharge_on_time", .7)]:   # ASSUMED signals
            noise = r.normal(0, .15, len(idx))
            if bias:                                    # injected bias 1: noisier and lower for shared-phone/document-poor
                bad = shared[idx] | docp[idx]
                noise = noise - .08 * bad + r.normal(0, .1, len(idx)) * bad
            F[nm][idx] = np.clip(base + .25 * np.tanh(cash) - C.ON_TIME_RISK_LOAD * z_risk[idx] + noise, 0, 1)
    F = pd.DataFrame(F)
    F["night_active_share"] = np.clip(.3 + (.055 * fem if bias else 0) + .1 * r.standard_normal(n), 0, 1)   # planted proxy: gender
    F["local_market_share"] = np.clip(.2 + (.1 * region if bias else 0) + .1 * r.standard_normal(n), 0, 1)  # planted proxy: region
    F.insert(0, "id", np.arange(n))
    F = F[["id"] + FEATURES]

    # ---- tables
    if cohort == "oot":
        split = np.full(n, "oot")
    else:
        split = r.choice(["train", "val", "test"], n, p=[C.SPLIT_FRAC["train"], C.SPLIT_FRAC["val"], C.SPLIT_FRAC["test"]])
    ids = np.arange(n)
    people = pd.DataFrame({
        "id": ids, "split": split, "gender": np.where(fem, "F", "M"), "urban_rural": np.where(rural, "rural", "urban"),
        "region": ["R%d" % (x + 1) for x in region], "occupation": [OCC[o] for o in occ],
        "thin_file": thin.astype(int), "shared_phone": shared.astype(int), "document_poor": docp.astype(int),
        "loan_amount": np.round(emi * 12), "loan_term": 12, "y_true": y, "r_true": r_true})
    latents = pd.DataFrame({"id": ids, "mu_income": mu, "sigma_income": sigma, "shock_prob": shock_p,
                            "rho_persist": rho, "buffer0": bal0})
    hist = [inc, out, bal]
    kin = keep
    full = [np.vstack([h, k_]).T.ravel() for h, k_ in zip(hist, kin)]
    monthly = pd.DataFrame({"id": np.repeat(ids, T + H), "month": np.tile(np.arange(1, T + H + 1), n),
                            "inflow": full[0], "outflow": full[1], "balance": full[2]})

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-12)
    legacy = -z(F.emi_to_income.values) + .3 * z(F.weeks_of_buffer.values)
    appr = legacy >= np.quantile(legacy, 1 - C.LEGACY_APPROVE_SHARE)
    explored = (~appr) & (r.random(n) < eps)
    observed = appr | explored
    lender = pd.DataFrame({
        "id": ids, "approved_legacy": appr.astype(int), "explored": explored.astype(int),
        "propensity": np.where(appr, 1.0, eps), "observed": observed.astype(int),
        "y_obs": np.where(observed, y, np.nan)})
    return {"people": people, "features": F, "monthly": monthly, "latents": latents, "lender_history": lender}
