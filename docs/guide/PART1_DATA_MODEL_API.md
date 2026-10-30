# Part 1: data, model, API

You own: `src/secondlook/{berka,sim,model,api}/`, `scripts/make_data.py`, `scripts/train.py`, `app/Home.py`, `app/pages/1_Applicant.py`, `app/pages/5_Data_and_Limits.py`, and tests `tests/test_p1_*.py`. You also lead the merge at 8:30 AM.

Other people are waiting on you for two handoffs. **Handoff A (real data tables) is the most urgent thing in the project**, so do steps 1 to 4 first and push. Work against `SL_DATA=data/fixtures/synth` until then.

Time budget: steps 1 to 4 about 2 h, 5 to 7 about 3 h, 8 to 10 about 3 h, Berka (step 11) only after everything else works.

---

## Step 0. Branch and setup

Follow `00_START_HERE.md` section 3, with branch `part1-data-model`. Create the package folders `src/secondlook/{sim,berka,api}/` each with an empty `__init__.py`.

## Step 1. Extend `config.py` (5 min)

Append to `src/secondlook/config.py` (these are the tuned simulator constants; the numbers in step 3 were measured with exactly these):

```python
# simulator (Part 1). Tuned so base default is about 11%, oracle AUC about 0.78, scorer AUC about 0.70 to 0.72
LTI_SCALE = 0.5          # loan instalment relative to income
POST_SHOCK_P = 0.25      # monthly chance of an unobserved expense shock after the loan
POST_SHOCK_MAX = 2.0     # shock size, multiples of monthly income (uniform 0.3 to this)
LOGIT_SLOPE = 0.55       # how strongly the simulated shortfall rate drives true risk
LOGIT_INTERCEPT = -1.0
LOGIT_FRAILTY_SD = 0.6   # unobserved heterogeneity; keeps the oracle AUC below 1
LEGACY_APPROVE_SHARE = 0.60
BIAS_FLAGS = True        # measurement bias, income-volatility gap, planted proxies (False = control run)
```

## Step 2. `model/features.py`: the one feature function (20 min)

Create `src/secondlook/model/features.py`. The simulator, Part 3 and the Berka code all call this, so features cannot drift apart.

```python
import numpy as np


def cash_features(inc, out, bal, emi):
    """inc, out, bal: arrays (K, w) months oldest first. emi: array (K,). Returns dict of arrays (K,)."""
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
```
Test (`tests/test_p1_features.py`): feed constant inflow 1000, outflow 800, balance 2000, emi 200 with K=2, w=6: expect `inflow_cv == 0`, `expense_to_income == 0.8`, `emi_to_income == 0.2`, `dip_months == 0`.

## Step 3. `sim/generate.py`: the simulator (about 1 h)

Create `src/secondlook/sim/generate.py` with the code below. It was run end to end on 20,000 people and gave: base default rate 10.8%, oracle AUC 0.78, plain logistic AUC 0.71, gender-proxy AUC 0.65, runtime under 1 second. Your job is to wrap it so it produces all the contract tables.

```python
import numpy as np
import pandas as pd

from .. import config as C
from ..model.features import cash_features

OCC = ["gig_driver", "gig_delivery", "shopkeeper", "student_parttime", "salaried_informal"]
P_OCC = np.array([.25, .20, .25, .10, .20])
MU = np.log([18000, 16000, 25000, 8000, 20000])      # monthly income level by occupation (assumption)
SIG = np.array([.30, .35, .28, .40, .15])            # income volatility by occupation (assumption)
RHO_A = np.array([2, 2, 3, 2, 4.]); RHO_B = np.array([3, 3, 3, 3, 3.])   # persistence ~ Beta(a, b)
SHOCK = np.array([.06, .07, .05, .08, .02])          # monthly dip probability by occupation
P_FEMALE = np.array([.12, .15, .40, .55, .45])


def generate(n, seed, cohort="main", T=C.HISTORY_MONTHS, H=C.OUTCOME_MONTHS, K=20, eps=C.EXPLORATION_EPS):
    """cohort: 'main' | 'oot' (drifted volatility and shocks) | 'null' (no bias, unequal base rates).
    Returns dict of DataFrames: people, features, monthly, latents, lender_history."""
    r = np.random.default_rng(seed)
    bias = C.BIAS_FLAGS and cohort != "null"
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
        if k == 0: keep = (i2, o2, b2)
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
            F[nm][idx] = np.clip(base + .25 * np.tanh(cash) + noise, 0, 1)
    F = pd.DataFrame(F)
    F["night_active_share"] = np.clip(.3 + (.055 * fem if bias else 0) + .1 * r.standard_normal(n), 0, 1)   # planted proxy: gender
    F["local_market_share"] = np.clip(.2 + (.1 * region if bias else 0) + .1 * r.standard_normal(n), 0, 1)  # planted proxy: region
    F.insert(0, "id", np.arange(n))
    # ... continue with the tables below
```

Now add, after that code, the table assembly (write it yourself following these exact rules; it is short):

1. **people**: columns `id, split, gender (F/M), urban_rural, region ("R1".."R6" = region+1), occupation (OCC[occ]), thin_file, shared_phone, document_poor (0/1 ints), loan_amount (= round(emi*12)), loan_term (12), y_true, r_true`. `split`: for cohort `main` and `null`, draw per person from `train/val/test` with probabilities 0.6/0.2/0.2 using `r.choice`; for cohort `oot` all `"oot"`.
2. **features**: `F` with column order `["id"] + contracts.FEATURES`.
3. **latents**: `id, mu_income=mu, sigma_income=sigma, shock_prob=shock_p, rho_persist=rho, buffer0=bal0`.
4. **monthly**: history `inc, out, bal` (shape 24 x n) concatenated with the first outcome run `keep` (12 x n) gives 36 x n. Build the long table: `id = np.repeat(np.arange(n), 36)`, `month = np.tile(np.arange(1, 37), n)`, and each of `inflow, outflow, balance` = `np.vstack([hist, outcome]).T.ravel()`.
5. **lender_history** (injected bias 4, selective labels): `legacy = -z(F.emi_to_income) + 0.3*z(F.weeks_of_buffer)` where `z` = standardise; `approved_legacy = legacy >= quantile(legacy, 1 - LEGACY_APPROVE_SHARE)`; `explored = (~approved_legacy) & (r.random(n) < eps)`; `observed = approved_legacy | explored`; `propensity = where(approved_legacy, 1.0, eps)`; `y_obs = where(observed, y, NaN)`. Output columns exactly as in the contract (`explored` and `observed` as 0/1 ints).

Return `{"people":..., "features":..., "monthly":..., "latents":..., "lender_history":...}`.

**Check**: in a REPL, `generate(20000, 1)["people"].y_true.mean()` must be between 0.09 and 0.13. If not, change only `LOGIT_INTERCEPT` by ±0.2 steps. A plain `LogisticRegression` on standardised features trained on 12,000 and tested on 8,000 should give AUC 0.68 to 0.76; the AUC of `r_true` against `y_true` (oracle) should be 0.74 to 0.82. If the scorer AUC is above 0.85, the data is too easy: raise `LOGIT_FRAILTY_SD` by 0.2. Do not spend more than 30 minutes tuning.

## Step 4. `scripts/make_data.py` and HANDOFF A (20 min)

```python
"""Writes data/synth (main + out-of-time) and data/synth_null."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from secondlook import config as C, contracts as K
from secondlook.sim.generate import generate

def write(tables, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if name in K.TABLES: K.check(name, df)
        df.to_parquet(out / f"{name}.parquet", index=False)

main = generate(C.N_PEOPLE, C.SEED, "main")
oot = generate(C.N_PEOPLE // 5, C.SEED + 1, "oot")
off = C.N_PEOPLE
for t in oot.values():                       # shift ids so they stay unique after concatenation
    t["id"] = t["id"] + off
merged = {k: pd.concat([main[k], oot[k]], ignore_index=True) for k in main}
# scores are written later by train.py. Remember: lender_history for the oot rows is also kept.
write(merged, C.ROOT / "data/synth")
write(generate(C.N_PEOPLE, C.SEED + 2, "null"), C.ROOT / "data/synth_null")
print("done")
```
Run: `~/.venv/bin/python scripts/make_data.py`. `make_data.py` does not write `scores.parquet`; `train.py` does (step 5). Parts 2 and 3 must not read `scores` from `data/synth` until Handoff B.

Commit and push. **Post in the team chat: "Handoff A done"** and say which tables exist. (`scores.parquet` arrives at Handoff B.)

## Step 5. `scripts/train.py` and HANDOFF B (45 min)

Upgrade the reference scorer in `model/scorer.py` only where listed; do not rename anything. The training script does:

1. Load `people`, `features`; use `split == "train"` to fit, `"val"` to calibrate and set tau, `"test"` and `"oot"` to report.
2. **Sign constraints**: keep all 17 features in the model at first (the proxy-leakage test needs the planted proxies there), then drop any feature whose fitted sign contradicts the expert direction. Features with direction 0 (the two planted proxies, `spike_months`) are never dropped. Loop:
   ```python
   feats = list(K.FEATURES)
   while True:
       s = Scorer(feats).fit(Xtr, ytr, Xva, yva)
       w = dict(zip(s.features, s.lr_.coef_[0]))       # coefficient on the standardised scale
       bad = [f for f in feats if K.FEATURE_DIRECTION[f] * w[f] < 0]   # wrong sign vs expert direction
       if not bad: break
       feats = [f for f in feats if f not in bad]
   ```
   Log which features were dropped to `artefacts/metrics.json`.
3. `s.set_tau_cost(Xva, yva)` (cost-based tau, c_default 5 x c_lost). If `config.TARGET_APPROVAL_RATE` is set, instead choose tau as the quantile of `p_default` on val giving that approval rate.
4. Write `scores.parquet` for ALL rows (all splits), columns `id, p_default, logit, decision, tau`; call `contracts.check`.
5. `s.config_hash = hashlib.sha1(json.dumps({k: str(v) for k, v in vars(config).items() if k.isupper()}, sort_keys=True).encode()).hexdigest()[:10]`; `save_model(s)`.
6. `artefacts/metrics.json`: AUC with 95% bootstrap CI (B=500) on test and on oot; Brier; ECE (10 bins); calibration by gender; dropped features; tau; approval rate on val.
7. Write `artefacts/data_dictionary.json` from `contracts.FEATURE_SOURCE` and `FEATURE_DIRECTION`; descriptions are one plain sentence per feature (write them by hand: e.g. `weeks_of_buffer`: "Average month-end balance divided by average weekly spending").

**Acceptance**: AUC on test between 0.68 and 0.78; OOT AUC lower than test AUC (that is the point: report it); ECE below 0.03 on val. Commit, push, post **"Handoff B done"**. Others now unset the fixture variables.

Optional booster (only if time remains): `xgboost.XGBClassifier(monotone_constraints=tuple(K.FEATURE_DIRECTION[f] for f in feats), max_depth=3, n_estimators=300, learning_rate=.05)`, report AUC beside the scorecard. If you add SHAP, call `shap.TreeExplainer(model).shap_values(X)` and store the top-5 stability (rerun with 5 seeds, count how often the top feature changes) in `metrics.json`. The main path never depends on the booster.

## Step 6. `model/explain.py` (30 min)

```python
import numpy as np
from ..contracts import FEATURE_DIRECTION

TEXT = {   # (feature, "raises risk"/"lowers risk") -> sentence; never mention gender, region, rural, phone, documents
    ("dip_months", "raises risk"): "Your income dipped well below its usual level in several recent months",
    ("dip_months", "lowers risk"): "Your income stayed steady month to month",
    ("weeks_of_buffer", "raises risk"): "Your savings cover only a few weeks of spending",
    ("weeks_of_buffer", "lowers risk"): "Your savings cover many weeks of spending",
    ("inflow_cv", "raises risk"): "Your monthly income varies a lot",
    ("inflow_cv", "lowers risk"): "Your monthly income is stable",
    ("emi_to_income", "raises risk"): "The instalment is large compared with your income",
    ("emi_to_income", "lowers risk"): "The instalment is small compared with your income",
    ("expense_to_income", "raises risk"): "You spend most of what you earn",
    ("expense_to_income", "lowers risk"): "You spend well under what you earn",
    ("low_balance_days", "raises risk"): "Your balance was low on many days",
    ("low_balance_days", "lowers risk"): "Your balance rarely ran low",
    ("rent_on_time", "raises risk"): "Rent payments were late more often (assumed signal)",
    ("rent_on_time", "lowers risk"): "Rent payments were on time (assumed signal)",
    # add the same pair for: mean_inflow, median_inflow, spike_months, balance_level, balance_slope,
    # utility_on_time, recharge_on_time, months_observed
}
SKIP = {"night_active_share", "local_market_share"}   # planted proxies are never shown as reasons


def reasons(scorer, x_row, top=5):
    """x_row: one-row DataFrame. Exact for the logistic scorecard: contribution_j = w_j * (x_j - mean_j)."""
    lp = scorer.linear_params()
    mean = dict(zip(scorer.features, scorer.mean_))
    out = []
    for f, w in zip(lp["features"], lp["w"]):
        if f in SKIP: continue
        c = float(w * (x_row[f].iloc[0] - mean[f]))
        d = "raises risk" if c > 0 else "lowers risk"
        out.append({"feature": f, "contribution": c, "direction": d, "text": TEXT.get((f, d), f"{f} {d}")})
    return sorted(out, key=lambda r: -abs(r["contribution"]))[:top]
```
Fill in every missing `TEXT` pair (no fallback text should ever reach the UI). Test: contributions of all features plus the intercept term reproduce `scorer.logit` for one row within 1e-6 (the sum over ALL features including skipped ones plus `b + w @ mean` constant).

## Step 7. Tests for the data and model (30 min)

`tests/test_p1_sim.py`, `test_p1_scorer.py`. Use `generate(3000, 1)` for speed. Assertions:
- same seed twice gives identical `people` (use `pd.testing.assert_frame_equal`);
- base default rate between 0.07 and 0.16 at n=3000;
- `id` unique in people/features/scores; every table passes `contracts.check`;
- no column named like a group or latent appears in `features`;
- `bias=False` (set `C.BIAS_FLAGS=False` with `monkeypatch`) gives gender proxy AUC between 0.45 and 0.55 (`roc_auc_score(people.gender=="F", features.night_active_share)`); with bias on it is between 0.58 and 0.72;
- scorer sign constraints: every kept feature has `FEATURE_DIRECTION * coef >= 0`;
- `scorer.logit_threshold()` consistent: `(logit <= thr) == (p_default <= tau)` for all rows;
- `lender_history`: `y_obs` is NaN exactly where `observed == 0`; `propensity` in (0, 1].

## Step 8. API `api/schemas.py`, `api/main.py` (1 h)

`schemas.py` (pydantic v2):
```python
from pydantic import BaseModel, Field
class ScoreRequest(BaseModel):
    person_id: int | None = None                       # use a stored synthetic person
    features: dict[str, float] | None = None           # or the observed features directly
class RecourseRequest(BaseModel):
    person_id: int
    horizon: int = Field(12, ge=1, le=24)
    target_prob: float = Field(0.8, gt=0, lt=1)
    rho_override: float | None = Field(None, ge=0, le=0.95)
```
`main.py`: create `app = FastAPI()`; load once at startup: `scorer = load_model()`, `features.parquet`, `people.parquet`, `scores.parquet`, `data_dictionary.json`. Every response includes `config_hash` and `disclaimer` ("Synthetic data; an audit-tool demonstration, not a lending decision tool."). Routes (table in spec section 8). Implementation rules:
- `/health` returns `{"status": "ok", "config_hash": ...}`.
- `/score`: if `person_id` given take the row from `features.parquet`, else build a one-row DataFrame from `features` after checking that all `FEATURES` are present (422 otherwise). Return `p_default`, `decision` ("approve"/"refer"/"decline": approve if p<=tau and the conformal set is a singleton, `refer` if abstain, else decline), `tau`, `reasons` (from `explain.reasons`), `confidence`, `thin_file`. For `confidence` try `from secondlook.uncertainty.conformal import load_conformal` inside a try/except `ImportError/FileNotFoundError`; if unavailable return `{"available": false}`.
- `/recourse`: try `from secondlook.recourse.plan import plan`; on ImportError raise `HTTPException(501, "not ready")`.
- `/audit/summary`, `/uncertainty/conformal`, `/uncertainty/poolfresh`: read the JSON from `ART_DIR` (`audit_summary.json`, `uncertainty.json`, `poolfresh.json`); 501 if the file does not exist. `/audit/nolabel` calls `nolabel_bounds` (501 if missing). `/data/dictionary`: the JSON file.
Test (`test_p1_api.py` with `fastapi.testclient.TestClient`): `/health` 200; `/score` with `person_id=0` returns `p_default` in [0,1] and 5 reasons; missing feature gives 422.

Run: `~/.venv/bin/uvicorn secondlook.api.main:app --port 8000`.

## Step 9. Streamlit pages (2 h)

All pages start with: `import sys; sys.path.insert(0, "src")`, `import streamlit as st`, `from secondlook import config as C`. Load data with `@st.cache_data`. Use plotly. Every page ends with `st.caption(disclaimer + " config_hash=" + hash)`.

**`app/Home.py`**: one paragraph pitch; a boxed "Real vs assumed" list: real = Berka bank data used for calibration; simulated = all people; assumed = rent, utility, recharge, planted proxies. Link to other pages via `st.page_link`.

**`app/pages/1_Applicant.py`** (call `load_model()` and `explain.reasons` directly, no API needed):
1. Consent block: bullet list of what data is used (cash-flow summaries only), how long (this session), a "Revoke consent" button that clears `st.session_state` and stops the page (`st.stop()`), and a ledger table in `st.session_state["ledger"]` (rows: time, purpose, data used) that gains a row each time the user scores someone.
2. Selectbox of synthetic people (show id, occupation, thin_file) or expander to edit 5 key features with sliders (`weeks_of_buffer`, `inflow_cv`, `dip_months`, `emi_to_income`, `expense_to_income`); other features keep the person's values.
3. Output: decision card, `p_default` as a gauge/metric, the top-5 reasons list, a slider for tau (default `scorer.tau`), and a confidence box: if conformal is loadable show "We are not sure about this application: referred to a person" for abstain, else hide.

**`app/pages/5_Data_and_Limits.py`**: data dictionary table from the JSON with a coloured `source` column; bullet list of injected biases (use the four numbered in `docs/IMPLEMENTATION_SPEC.md` 2.2.6); metrics from `metrics.json` (AUC with CI on test and oot, dropped features); Berka sanity results if the file exists (step 11); a "claims we do not make" list copied from `docs/PROJECT_PLAN.md`.

## Step 10. Merge lead duties (8:30 AM)

Follow `00_START_HERE.md` section 8. You are responsible for the merged repo starting in a clean clone using only the README commands.

## Step 11. Berka (only after steps 1 to 10 work; cut first if short on time)

Purpose: a real-data sanity run and calibration targets. The local file is `data/berka/financial.sql` (MariaDB dump).

`berka/load.py`:
```python
import ast, re
import pandas as pd

def table(path, name):
    """Parse INSERT statements for one table out of a mysqldump file."""
    cols, rows = None, []
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(rf"CREATE TABLE `{name}` \((.*?)\n\)", text, re.S)
    cols = re.findall(r"^\s*`(\w+)`", m.group(1), re.M)
    for ins in re.finditer(rf"INSERT INTO `{name}` VALUES (.*?);\n", text, re.S):
        payload = ins.group(1).replace("NULL", "None")
        rows.extend(ast.literal_eval("[" + payload + "]"))
    return pd.DataFrame(rows, columns=cols)
```
Check on `loan`: 682 rows; `loan.status.value_counts()` shows A, B, C, D. Defaults = `status in ("B","D")` (expect 76). Check the actual values of `trans.type` and `trans.date` format before using them: print `trans.type.unique()` and `trans.head()`.

`berka/features.py`: for each loan, take that account's transactions with `date < loan.date` in the last 6 months; build monthly arrays: inflow = sum of credit amounts, outflow = sum of debit amounts, balance = last balance in the month; then call `cash_features(inc[None], out[None], bal[None], emi=[loan.payments])`. **Never use a transaction on or after the loan date** (a test asserts it). Skip loans with fewer than 3 months of history.

Sanity run: fit the same scorecard (sign-constrained logistic) on Berka features with 5-fold cross-validated AUC and a bootstrap 95% CI; report gender and district-tercile approval rates. Write `artefacts/berka_sanity.json` (`{"n": 682, "defaults": 76, "auc": [lo, est, hi], "window_months": 6}`). Say on the Data page: with 76 defaults the CI is wide and Berka is not Indian data.

Calibration targets (`berka/calibrate.py`): compute Berka's inflow-CV median and quantiles, dip-month frequency, weeks-of-buffer quantiles; print a comparison table against the simulator's. If they differ by more than a factor of 2, adjust `SIG` (volatility) and `fixed` ranges and note the final values in the datasheet. Do not tune more than 30 minutes.

`tests/test_p1_berka.py::test_temporal`: for every loan, max transaction date used < loan date.
