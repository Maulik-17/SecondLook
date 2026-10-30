# Part 3: recourse and submission

You own: `src/secondlook/recourse/`, `scripts/run_recourse.py`, `app/pages/3_Recourse.py`, `tests/test_p3_*.py`, and the submission material: `README.md`, `docs/MODEL_CARD.md`, `docs/DATASHEET.md`, the demo script and the video. Branch `part3-recourse`.

You read (from Part 1): `people`, `features`, `monthly`, `scores` parquet files, `latents` (only for tests on the real run) and `load_model()`, plus `model.features.cash_features`. Start now with fixtures (`00_START_HERE.md` section 3). Never edit Part 1's or Part 2's files.

Time plan: steps 1 to 5 about 4.5 h, step 6 (page) 1.5 h, step 7 (tests) 1 h, then the submission material (steps 8 to 10) from 8:30 AM. If you are behind, finish steps 1 to 5 basic versions first (single stress test, no rho slider) and start step 8 early in parallel: README text does not depend on the code.

Create `src/secondlook/recourse/__init__.py` (empty).

## What recourse means here

A rejected applicant asks: "what should I do over the next H months so that I am approved?" Income is volatile, so we do not give one fixed instruction. We simulate many possible futures of their income, apply a candidate behaviour plan, rescore every month with the same model, and report the **probability** of reaching approval within H months. We pick the cheapest plan whose probability is at least the target (default 0.8). This is a model-conditional prediction on synthetic data, not a promise.

---

## Step 1. `recourse/income_process.py` (1 h)

Inputs: the person's monthly history from `monthly.parquet` (months 1..24), their stored feature row (to know `months_observed = w`), and occupation from `people.parquet`.

**Use only the last `w` months** of history (`w = months_observed`, 3 to 5 for thin files). Months before that are hidden from the model and must be hidden from recourse too.

Model: `x_t = m + rho (x_{t-1} - m) + sigma z_t` on `x = log(inflow)`, `z ~ N(0,1)`.

```python
import numpy as np, pandas as pd
from .. import config as C

def fit_ar1(x):
    """Closed-form fit on a short log-income series. Returns dict(m, rho, sigma, e_last, se_rho, w)."""
    w = len(x); m = x.mean(); d = x - m
    rho = float(np.clip((d[1:] * d[:-1]).sum() / max((d[:-1] ** 2).sum(), 1e-9), 0.0, 0.9)) if w > 2 else 0.0
    resid = d[1:] - rho * d[:-1] if w > 2 else d
    sigma = float(max(resid.std(ddof=1) if len(resid) > 1 else 0.2, 0.03))
    return {"m": float(m), "rho": rho, "sigma": sigma, "e_last": float(d[-1]), "se_rho": (1 - rho ** 2) / np.sqrt(max(w, 2)), "w": w}

def population_prior(monthly, people):
    """Fit every person on all 24 history months; return mean and sd of (rho, log sigma) per occupation."""
    h = monthly[monthly.month <= C.HISTORY_MONTHS]
    rows = []
    for pid, g in h.groupby("id"):
        f = fit_ar1(np.log(g.sort_values("month").inflow.values))
        rows.append((pid, f["rho"], np.log(f["sigma"])))
    d = pd.DataFrame(rows, columns=["id", "rho", "logsig"]).merge(people[["id", "occupation"]], on="id")
    return d.groupby("occupation").agg(rho_mean=("rho", "mean"), rho_sd=("rho", "std"),
                                       ls_mean=("logsig", "mean"), ls_sd=("logsig", "std")).to_dict("index")
```
(`population_prior` loops over 20,000 people; compute it once, cache it with `joblib` in `ART_DIR/income_prior.joblib`, and only recompute if the file is missing. On fixtures use a 2,000-person subsample.)

**Shrinkage for short series**: with `lam = w / (w + 8)`, `rho_post = lam*rho_hat + (1-lam)*prior_rho_mean`, `logsig_post = lam*log(sigma_hat) + (1-lam)*prior_ls_mean`. 

**Parameter draws per Monte Carlo path** (call `draw_params(fit, prior, K, rng, rho_override=None)`):
- `rho = clip(N(rho_post, (se_rho*(1-lam) + 0.05)^2), 0, 0.95)` unless `rho_override` is given (then all paths use it);
- `sigma = exp(logsig_post + N(0, 1/sqrt(2*max(w-1,1))))`;
- `m = N(m_hat, sigma^2 / w)` (uncertainty in the mean level);
- the starting residual is `e_last` (the last observed deviation from the mean), carried forward by the AR recursion: with `rho = 0` it is forgotten at once, with `rho` near 0.9 it persists. **This is the persistence slider.**
Also estimate a per-month dip probability `p_dip = clip(dip_months / w, 0.01, 0.2)` from the person's stored `dip_months` feature; each simulated month has chance `p_dip` of a one-month dip of `-U(0.25, 0.6)` in log income.

**Outflow model** (needed to keep balances consistent): OLS of `outflow` on `inflow` over the same window: `out = a*inc + f`, clip `a` to [0.3, 0.95] and `f` to >= 0; use `balance_0 = last observed balance`.

Test: simulate a long AR(1) with known `rho = 0.5, sigma = 0.2`, fit with `w = 200`: estimates within 0.1 and 0.03. With `w = 3`, `rho_post` is within 0.1 of the prior mean (strong shrinkage).

## Step 2. `recourse/actions.py` (20 min)

Four actions, three levels each (0 = do nothing). Costs are assumptions, in "effort points per month", kept in `config.py`-style constants in this file and shown on the page.
```python
ACTIONS = {
    "cut_spend":    {"levels": [0.00, 0.05, 0.10], "cost": [0.0, 1.0, 2.5]},   # share of variable spending cut
    "extra_income": {"levels": [0.00, 0.05, 0.10], "cost": [0.0, 2.0, 4.5]},   # extra inflow, share of mean income
    "on_time":      {"levels": [0.00, 0.10, 0.20], "cost": [0.0, 0.5, 1.0]},   # rise in rent/utility/recharge on-time rate (ASSUMED signal)
    "smaller_loan": {"levels": [0.00, 0.15, 0.30], "cost": [0.0, 0.8, 2.0]},   # one-off cut in requested loan, felt as lost utility
}
def plan_cost(levels_idx, horizon):
    """levels_idx: dict action -> 0/1/2. Monthly costs times horizon, except smaller_loan which is one-off."""
```
Costs: `cost = sum over actions != smaller_loan of cost[idx] * horizon + cost_smaller_loan[idx]`. `all_plans()` returns the 81 combinations `itertools.product(range(3), repeat=4)`.

**How actions act (through the features, gated by cash):** `cut_spend` multiplies the variable part of outflow: `out_t = a*inc_t*(1 - cut) + f`; `extra_income` multiplies inflow: `inc_t *= 1 + g`; `on_time` raises the three `*_on_time` features: `min(1, value + uplift)`; `smaller_loan` multiplies the instalment `emi *= 1 - l`. A balance can only grow from actual surplus: `bal_t = max(bal_{t-1} + inc_t - out_t, floor)` with `floor = -0.3*mean_inflow`; you cannot hold a balance you do not have.

## Step 3. `recourse/montecarlo.py` (1.5 h)

```python
def simulate(person, fit, prior, plan_idx, horizon, K, tau, scorer, seed, rho_override=None, vol_mult=1.0, on_time_effective=True):
    rng = np.random.default_rng(seed)                       # SAME seed for every plan: common random numbers
    Z = rng.standard_normal((K, horizon)); U = rng.random((K, horizon)); D = rng.uniform(.25, .6, (K, horizon))
    P = rng.standard_normal((K, 3))                         # parameter noise
    ...
```
Pseudocode to implement:
1. Draw parameters for K paths from step 1 using the pre-drawn noise `P` (so they are also common across plans).
2. For month `t = 1..H`: `e = rho*e + sqrt(1 - rho^2) * sigma*vol_mult * Z[:, t-1]` (start `e = e_last`); dip: `dip = where(U[:, t-1] < p_dip, -D[:, t-1], 0)`; `inc_t = exp(m + e + dip) * (1 + extra_income)`; `out_t = a*inc_t*(1 - cut) + f`; `bal_t = max(bal_{t-1} + inc_t - out_t, floor)`.
3. Rolling window: the window used for the features at review month `t` is the last `min(w + t, 24)` months of `concat(history_window, simulated months 1..t)`; `history_window` is the person's last `w` real months (arrays of shape `(K, w)` repeated K times).
4. Features at review `t`: `cash_features(inc_win, out_win, bal_win, emi_array)` from Part 1 where `emi = person_emi * (1 - smaller_loan)` and `person_emi = loan_amount / loan_term`. Build a DataFrame with ALL columns of `contracts.FEATURES`: the cash ones from the function, the three `*_on_time` from the stored person row plus the uplift (clipped to 1; skip uplift when `on_time_effective=False`), the two planted proxies copied from the stored row. Score `p = scorer.predict_proba(df)`.
5. `approved[:, t] = p <= tau`. `first[k] = min t with approved` (or `inf`). Success = `first <= horizon`.
6. Return `{"success": bool array (K,), "first": array (K,), "p_matrix": (K, H), "min_buffer_weeks": array}` where `min_buffer_weeks` is the minimum over months of `bal / (mean_out / 4.345)`.
Performance target: one call with K=1000, H=12 under 0.3 s (all numpy, one `predict_proba` call per month).

Checks: with `plan = (0,0,0,0)`, `rho_override = 0`, `vol_mult = 0`, and `Z = 0` the path is deterministic; success matches whether the deterministic projection is approved. The same seed gives identical results on repeated calls.

## Step 4. `recourse/plan.py` (1 h)

```python
def plan(person_id, horizon=12, target_prob=0.8, rho_override=None, tau=None):
    # 1 load person (cache tables in module-level variables), scorer, prior; tau = tau or scorer.tau
    # 2 for each of the 81 plans sorted by cost ascending:  K=500 paths -> p = success.mean()
    #       stop at the first plan with p >= target_prob   -> chosen
    #   if none reaches the target: chosen = plan with max p, infeasible = True
    # 3 re-evaluate the chosen plan with K=1000 and compute the contract output
```
Output (exactly the contract in `docs/SPLIT_CONTRACT.md`):
- `plan`: list of `{"action": name, "amount": level value, "cost": cost}` for actions with idx > 0 (empty list if already ok);
- `cost`, `p_success = wilson(k, K)` as `[lo, est, hi]` (copy the `wilson` function from Part 2's guide into `recourse/stats.py`; do not import Part 2);
- `months_to_approval_quantiles`: `p10, p50, p90` of `first` across all paths; `None` when the quantile is infinite;
- `fan_chart`: for months 1..H the 10th, 50th, 90th percentile of `p_matrix` over paths;
- `infeasible: bool` (extra key, allowed);
- `fragile` and `stress` from step 5.
Edge cases: a person already approved (`p_default <= tau` now) returns an empty plan, `cost = 0`, `p_success = [1,1,1]`; unknown id raises `KeyError` (the API turns it into 404).

## Step 5. `recourse/stress.py` (1 h)

`stress_tests(person, chosen_plan, ...)` returns `{"drift", "threshold_shift", "gaming", "persistence"}`, each the success probability (K=1000, same seed) under:
- `drift`: `vol_mult = 1.2` (income volatility 20% higher than fitted);
- `threshold_shift`: `tau - 0.02` (stricter lender);
- `gaming`: `on_time_effective=False` (the plan's gain from the assumed rent/utility/recharge signals is removed; shows how much of the plan relies on assumed signals);
- `persistence`: the minimum over `rho_override in (0.0, 0.45, 0.9)`.
`fragile = (base p_success >= target) and (min(stress values) < 0.6)`. The full retraining stress test in the spec is optional; if you add it, add a fifth key `retrain`.

`run_recourse_by_group(n_per_group=100)` (`scripts/run_recourse.py` calls it): for each (attribute, level) among the 7 attributes, sample up to `n_per_group` people with `split == "test"` and `decision == 0` (rejected), evaluate **a moderate fixed plan** `(1,1,1,1)` with K=300, take each person's `p50` months-to-approval, and write
`{"groups": [{"attribute", "level", "n", "median_months": median of p50 or None, "never_share": share with p50 None}]}` to `ART_DIR/recourse_by_group.json`. It takes a few minutes; run it with `n_per_group=30` while developing. **Tell Part 2 when this file exists**; they re-run their audit to pick it up.

## Step 6. `app/pages/3_Recourse.py` (1.5 h)

Begins with `import sys; sys.path.insert(0, "src")`, imports `plan` directly (offline mode). Layout:
1. Select a rejected person from a dropdown (people with `decision == 0` in the test split; show id, occupation, thin_file), show current `p_default` and tau.
2. Sliders: horizon (1 to 24, default 12), target probability (0.5 to 0.95, default 0.8), persistence `rho` (0.0 to 0.9, plus a checkbox "use fitted value" on by default).
3. Button "Find cheapest plan". Show: the plan table (action, amount, cost), total cost, success probability with interval, months-to-approval quantiles, a message if `infeasible`.
4. Fan chart: plotly with a shaded band between `q10` and `q90`, the median line, and a horizontal line at tau.
5. Stress-test table (four rows) and a red "fragile" badge when applicable.
6. Group view: months-to-approval by group from `recourse_by_group.json` (bar chart, with `n`).
7. Footer: the disclaimer, plus: "This is a model-based probability on synthetic data, not a guarantee. The rent, utility and recharge signals are assumed." Add the model-extraction note: "This tool reveals how the score reacts to behaviour; real deployments must consider misuse."
Use `@st.cache_data` on `plan(...)` keyed by its arguments. A call must finish in under 5 s.

## Step 7. Tests (1 h), `tests/test_p3_recourse.py`

- success probability is non-decreasing in horizon (H=6 versus 12, same seed);
- common random numbers: two calls with the same arguments return identical results;
- monotonicity in intensity: plan (2,2,2,2) has success probability >= plan (0,0,0,0) (up to 0.02 Monte Carlo slack);
- infeasible target (0.999) returns the maximum-probability plan with `infeasible = True`;
- an already-approved person returns an empty plan;
- the contract keys exist and `p_success` is `[lo, est, hi]` with `lo <= est <= hi`;
- `fan_chart` arrays have length `horizon`;
- `rho_override = 0` and `0.9` give different `p_success` for at least one test person (the slider matters).

---

## Step 8. README (from 8:30 AM, about 1 h; write the skeleton earlier)

The portal requires: public repo, README with problem statement, stack and team, and a 3-minute video. Put the README at the repo root. Sections, in this order (fill every one; keep them factual and short):

1. Title, one-line pitch: "SecondLook: audit-first credit scoring for people with thin credit files". Status line: "Audit-tool demonstration on synthetic data. Not a lending decision tool."
2. Problem statement 3.4 (paste the text from the problem statement PDF: gig workers, shopkeepers and students lack credit history; build a score from rent, utility, recharge and cash-flow signals; fairness checks across groups; synthetic data with consent-based design; bonus: bias audit panel).
3. What SecondLook adds: the four numbered questions (burden, uncertainty, volatility-aware plan, no-label bounds).
4. Screenshots: two or three PNGs from the app (`docs/img/`).
5. Data: simulator (hidden ground truth), Berka calibration, what is real vs assumed table (copy from `docs/SPLIT_CONTRACT.md` and the data dictionary), the four injected biases.
6. Architecture diagram (ASCII is fine, copy from the existing README).
7. How to run, exactly:
   ```
   git clone <repo> && cd SecondLook
   bash scripts/fetch_berka.sh        # optional: downloads the Berka dump
   python scripts/make_data.py && python scripts/train.py
   python scripts/run_recourse.py && python scripts/run_audit.py
   streamlit run app/Home.py
   uvicorn secondlook.api.main:app --port 8000
   pytest -q
   ```
   (Use the real python path the team uses; write plain `python` for the public README and list required packages: numpy, scipy, pandas, pyarrow, scikit-learn, xgboost, statsmodels, fastapi, uvicorn, streamlit, plotly, pydantic, joblib, pytest; optional shap.) Confirm the commands work in a fresh clone before submitting.
8. Stack: scikit-learn, FastAPI, Streamlit, plotly, XGBoost (optional), SHAP (optional).
9. Team: Harshit Jangir (leader), Nikhil Saini, Shreya Chaudhary, Maulik Pratap Singhal. Event: SPHINX'26 24-hour hackathon, MNIT Jaipur, team NexusNavigators.
10. Limits and claims we do not make: data is synthetic or Berka (Czech, not Indian); rent/utility/recharge signals assumed; no validation on Indian thin-file borrowers; burden and effort are diagnostics not targets; the four-fifths rule has no legal force in India; no compliance claim for RBI FREE-AI or DPDP; no claim that borrowers or loan officers understand reason codes or conformal sets; post-processing fairness mitigation is not implemented.
11. Attribution: scikit-learn, XGBoost, FastAPI, Streamlit, plotly; Berka: P. Berka, PKDD'99 discovery challenge, obtained from the CTU Prague Relational Learning Repository; Motl and Schulte, "The CTU Prague Relational Learning Repository", arXiv:1511.03086. Methods adapted (we reimplemented them, copied no code): Fang et al. 2607.28847, Barrainkua et al. 2509.04128, De Toni et al. 2410.08007, Plecko 2605.11373, Kallus, Mao and Zhou 1906.00285, Tibshirani et al. (conformal under covariate shift), Scarone and Baeza-Yates 2606.18479. Before submitting, spot-check each arXiv id against the paper title.
12. Licence line (choose MIT unless the team says otherwise) and the disclaimer.

## Step 9. Model card and datasheet (45 min each)

`docs/MODEL_CARD.md` headings: Model details (logistic scorecard, sign constraints, Platt calibration, cost-based tau, version = `config_hash`); Intended use (research and demonstration of audit methods); Out-of-scope use (real lending decisions); Factors (groups audited); Metrics (read the numbers from `artefacts/metrics.json`: test AUC with CI, out-of-time AUC, ECE, group table highlights); Data (pointer to datasheet); Ethical considerations (selective labels, proxies, assumed signals, burden as diagnostic); Caveats (synthetic circularity: any bias found is a property of the injected generator, not of the world).

`docs/DATASHEET.md` headings: Motivation; Composition (tables and columns from the contract, sizes, splits, out-of-time cohort); Collection process (generator mechanism in 4 sentences: income process, shocks, EMI shortfall default, frailty); Injected biases (the four, with parameters); What is assumed vs calibrated vs real (Berka facts: 682 loans, 76 not successful, Czech); Preprocessing (features, temporal rule for Berka); Uses (and not); Distribution (generator code only, Berka not redistributed); Maintenance (none).

## Step 10. Demo script and video (9:45 to 11:15 AM)

The video must be under 3 minutes. Rehearse once with a timer, record in one or two takes. Screen recording: GNOME screen recorder (Ctrl+Alt+Shift+R) or OBS if installed; record at 1080p, microphone on, browser zoom 110%. Run the app on real data before recording and keep the browser on the Streamlit app.

| Time | Screen | Say |
|---|---|---|
| 0:00 to 0:20 | Home | The problem: 3.4, people with no credit history get rejected. A plain score is easy; the hard question is who carries the cost of fixing a rejection. All data synthetic or Czech Berka; this is an audit-tool demo. |
| 0:20 to 0:50 | Applicant | Consent screen, pick a gig worker, score, plain-language reasons, "we are not sure, referred to a person". |
| 0:50 to 1:35 | Audit | Group table with intervals; burden among the creditworthy rejected: worst group highlighted; drag tau: level-down warning "fairer by rejecting more is not fairer"; risk-matched gap. |
| 1:35 to 2:05 | Audit (proxy and no-label) | Proxy can be reconstructed but equal-risk gap small; no-label bounds are wide: a weak proxy tells you little. |
| 2:05 to 2:35 | Recourse | A rejected applicant; cheapest plan with 80% chance in 12 months; fan chart; fragile badge; persistence slider moves the probability. |
| 2:35 to 2:55 | Uncertainty and Data | Coverage and referral gaps shown together; exploration slider; data page labels assumed vs real. |
| 2:55 to 3:00 | Home | Limits stated; thank you. |

Upload the video where the portal asks (file or link; if a link, make it viewable without login). Save a copy in `docs/demo.mp4` only if under 50 MB, otherwise do not commit it.

## Step 11. Submission checklist (11:45 AM to 12:30 PM; portal opens 12:00 PM)

- [ ] Repo is public; clone it into a fresh folder and run the README commands.
- [ ] README complete (all 12 sections); screenshots present.
- [ ] `pytest -q` passes in the fresh clone.
- [ ] No raw Berka, no `data/synth`, no secrets committed; `git log` has no AI tool mentions.
- [ ] `scripts/fetch_berka.sh` documented in the README.
- [ ] Video under 3 minutes, plays without login.
- [ ] Team ID filled in the portal (still unknown: ask a volunteer at the desk).
- [ ] Submitted before 1:30 PM; screenshot of the confirmation.

## Step 12. Pitch preparation (after submission, 2:45 PM pitching)

Judged on originality, understanding, viability. Prepare 60-second answers to: What is new compared with a plain cash-flow score? (audit-first, burden on the creditworthy rejected, volatility-aware plan, honest bounds when labels are missing). Why synthetic data? (no access to lender data; consent-based design; we label everything assumed). Can this be gamed? (gaming ratio, assumed signals, extraction note). What would you do with real data? (replace the generator, keep the audit). Why is conformal abstention useful? (human in the loop on thin files). What are the limits? (the list in README section 10). Rehearse once.
