# Part 2: audit and uncertainty

You own: `src/secondlook/audit/`, `src/secondlook/uncertainty/`, `scripts/run_audit.py`, `app/pages/2_Audit.py`, `app/pages/4_Uncertainty.py`, `tests/test_p2_*.py`.

You never edit Part 1's or Part 3's files. You read these (all produced by Part 1; fixtures have the same format): `people.parquet`, `features.parquet`, `scores.parquet`, `lender_history.parquet`, and `load_model()`. Start now with `SL_DATA=data/fixtures/synth SL_ART=data/fixtures/artefacts` (see `00_START_HERE.md` section 3); branch `part2-audit`.

Conventions used below: `y = 1` means the person defaults; "good" means `y_true = 0` (would repay); "approved" means `scores.decision == 1`; "rejected" means `decision == 0`. A "group level" is one (attribute, level) pair such as ("gender", "F"). The audited attributes are the 7 keys of `contracts.GROUPS`. Levels with fewer than `config.MIN_GROUP_N` (50) people are reported but greyed out and excluded from "worst group".

Create the folders `audit/` and `uncertainty/` each with an empty `__init__.py`.

---

## Step 1. `audit/data.py`: one loader used by everything (15 min)

```python
import pandas as pd
from .. import config as C
from ..contracts import GROUPS, FEATURES

def load():
    d = C.DATA_DIR
    ppl = pd.read_parquet(d / "people.parquet")
    F = pd.read_parquet(d / "features.parquet")
    S = pd.read_parquet(d / "scores.parquet")
    LH = pd.read_parquet(d / "lender_history.parquet")
    df = ppl.merge(F, on="id").merge(S, on="id").merge(LH, on="id")
    return df          # one row per person; columns of every table

def levels():
    return [(a, l) for a, ls in GROUPS.items() for l in ls]

def mask(df, attr, level):
    return df[attr] == level
```
Evaluation population = `df[df.split == "test"]` unless a step says otherwise (the scorer was trained on `train`, calibrated on `val`).

## Step 2. `audit/metrics.py`: group table with confidence intervals (1 h)

Helpers:
```python
import numpy as np

def wilson(k, n, z=1.96):
    if n == 0: return [np.nan, np.nan, np.nan]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(p), float(c + h)]       # [lo, est, hi]

def boot(stat, df, B=1000, seed=0):
    """Percentile bootstrap of stat(df) -> float. Returns [lo, est, hi]."""
    r = np.random.default_rng(seed); n = len(df); vals = []
    for _ in range(B):
        vals.append(stat(df.iloc[r.integers(0, n, n)]))
    vals = np.array(vals)
    return [float(np.nanpercentile(vals, 2.5)), float(stat(df)), float(np.nanpercentile(vals, 97.5))]

def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))
```
`group_table(df, tau, B)` returns a list of dicts exactly as `groups` in `docs/SPLIT_CONTRACT.md`. For each group level `g` (rows `sub`):
- `n = len(sub)`
- `approval = wilson(approved.sum(), n)` where `approved = sub.p_default <= tau`
- `base_rate = wilson(sub.y_true.sum(), n)` (true default rate in the group)
- "among approved" metrics (the contract requires this label): restrict to `sub[approved]` for `ppv` = share of approved who are good = `wilson(good_approved, n_approved)`;
- `tpr` = P(approved | good) = `wilson(approved & good, good)`, `fpr` = P(approved | bad) = `wilson(approved & bad, bad)` (here "positive" means approved);
- `ece` of `p_default` vs `y_true` within the group; `brier = mean((p - y)^2)`;
- `missing_outcome_share = 1 - observed.mean()` from `lender_history.observed` (the share of the group with no real-world outcome).
Use `wilson` for proportions, `boot` only for burden metrics (it is slow). If `bootstrap=False` skip `boot` and return `[est, est, est]`.

Also compute the pooled **disparate-impact ratio** = min/max of group approval rates over one attribute at a time, and put it on the page with the note "The four-fifths rule is US employment guidance and has no legal force in India."

Tests (`test_p2_metrics.py`): `wilson(50, 100)` returns lo < 0.5 < hi and lo about 0.404; `boot` interval of the mean of a fixed normal sample covers the true mean in at least 90 of 100 repeated small simulations (use B=200).

## Step 3. `audit/burden.py`: burden on the creditworthy rejected (the headline, 2 h)

Idea: for a rejected person, the score has to fall from `s_i` to the threshold `s*`. The cheapest change moves the actionable features along the direction of steepest score decrease, where each feature has a cost weight. Applies to the logistic scorecard exactly.

**Actionable features and cost weights** (assumptions; ship three sets and show all three on the page). Create `audit/weights.py`:
```python
ACTIONABLE = ["weeks_of_buffer", "inflow_cv", "dip_months", "low_balance_days", "expense_to_income",
              "balance_level", "balance_slope", "rent_on_time", "utility_on_time", "recharge_on_time"]
# everything else is immutable (months_observed, emi_to_income, mean/median inflow, spike_months, planted proxies)
WEIGHTS = {   # higher = harder to change, in effort per standardised unit
    "base":      dict(weeks_of_buffer=2, inflow_cv=3, dip_months=3, low_balance_days=1.5, expense_to_income=2,
                      balance_level=2, balance_slope=1.5, rent_on_time=1, utility_on_time=1, recharge_on_time=0.5),
    "income_hard": dict(weeks_of_buffer=3, inflow_cv=6, dip_months=6, low_balance_days=2, expense_to_income=2,
                      balance_level=3, balance_slope=2, rent_on_time=1, utility_on_time=1, recharge_on_time=0.5),
    "behaviour_hard": dict(weeks_of_buffer=2, inflow_cv=2, dip_months=2, low_balance_days=3, expense_to_income=4,
                      balance_level=2, balance_slope=3, rent_on_time=3, utility_on_time=3, recharge_on_time=2),
}
```
**Formula** (work in standardised units: divide each raw weight `w_j` by the feature's training std `sd_j`, i.e. `v_j = w_j * sd_j`; only actionable `j`):
- `slack_i = s_i - s*` for rejected `i`, where `s_i = scorer.logit(x_i)`, `s* = scorer.logit_threshold()` for the chosen `tau` (see the note below on tau).
- direction cost: `denom = sqrt( sum_j v_j^2 / W_j )`
- `effort_i = slack_i / denom`
- the move itself: `delta_j = slack_i * (v_j / W_j) / sum_k (v_k^2 / W_k)` in standardised units; new raw feature `x_j' = x_j - delta_j * sd_j` (reduces the score).
- **Feasibility**: clip each `x_j'` to the 1st-99th percentile range of the training data for that feature (and rates to [0,1]). Recompute `s(x')`. If `s(x') > s*`, then `effort_i = inf` and the person counts toward `infeasible_share`.
Implementation tip: vectorise over all rejected people with numpy; `wp = scorer.linear_params()`; use `wp["w"][j]` for the raw weights.

**Using a different tau:** `s*` is a function of tau: `s* = logit_threshold()` is only valid for `s.tau`. For the level-down sweep compute `s*(tau)` by inverting the Platt calibration yourself: `s* = (logit(tau) - cal.intercept_) / cal.coef_[0,0]` with `cal = scorer.cal_`. Wrap this as `threshold_for(scorer, tau)` in `burden.py` and test that `threshold_for(s, s.tau) == s.logit_threshold()`.

**Outputs** (use rejected AND good, `decision == 0 and y_true == 0`; call this set `RG`):
- `burden` for group g = `sum(effort_i over RG ∩ g, finite only) / |g|`, with a bootstrap interval of the same statistic; `infeasible_share` = infeasible people in `RG ∩ g` / `|RG ∩ g|`. Also store `n`.
- `burden_headline`: the worst group (largest burden among levels with n >= 50), `ratio` = worst / best.
- `budget_curve` for g: for budgets `b` on a grid (`np.linspace(0, q95, 25)` where `q95` is the 95th percentile of finite efforts) the share of `RG ∩ g` with `effort <= b`. Output `budget` and `share` arrays.
- `risk_matched_gap`: bin ALL rejected people into 5 bins by true risk `r_true` (quintiles); within each bin compare mean finite effort between each level of an attribute and the rest; output the gap with a bootstrap interval. If the gap is near zero after matching, the raw burden gap is explained by risk; if it persists it is not. (Note: matching on the model's own risk would give zero by construction for a logistic model, so use `r_true`.)
- Sensitivity: run all of the above under all three weight sets; the page shows whether the worst group's identity and the sign of the gap change (`sign_stable: bool`).

Tests (`test_p2_burden.py`): (1) for a 3-feature toy logistic model, the closed-form effort equals a numerical minimum found with `scipy.optimize.minimize` (constraint `s(x') <= s*`, objective the weighted norm) to within 1e-3; (2) a person whose slack is 0 has effort 0; (3) fixing `W` equal for all features and standardising gives `effort = slack / ||v||`; (4) infeasible detection: a person with `slack` of 100 gets `inf`.

## Step 4. `audit/failure.py`: the failure-mode panel (45 min)

- **Level-down**: for tau in `np.linspace(0.03, 0.5, 20)`: approval rate (test split) and the worst-group burden (via `threshold_for`). Return the list `level_down` for the Pareto chart. Flag `level_down = True` when moving from the current tau to a lower-approval tau (approval down by more than 5 points) reduces the worst/best burden ratio by more than 10%: then "fairer" is being reached by rejecting more people.
- **Inversion**: the recommended direction for feature `j` is `-delta_j`; count rejected people for whom a recommended change moves a feature the wrong way versus `contracts.FEATURE_DIRECTION` (e.g. recommends lowering `weeks_of_buffer`). For a sign-constrained scorecard this is 0; report the count and set `inversion = count > 0`.
- **Gaming ratio**: share of the total score reduction (`sum_j w_j * delta_j*sd_j`) that comes from features whose source is `assumed` (the three `*_on_time`). Flag `gaming = ratio > 0.3`. Say on the page that these signals are assumed and not shown to cause default.
- **Infeasible share and tail cost**: overall infeasible share and the 95th percentile effort.
- **Referrals counted as burden**: alternative view of the burden table where abstained people (from the conformal model, Step 9) are added to the rejected set. Skip if the conformal model is not yet available.
- Static text: "The recourse endpoint reveals score sensitivities; this is a model-extraction risk."

## Step 5. `audit/proxy.py`: proxy-leakage card (45 min)

For `A` in gender, urban_rural, region (test + val rows, features = `scorer.features` only):
1. Cross-fitted AUC: `cross_val_predict(LogisticRegression(max_iter=1000), Xz, A_binary, cv=5, method="predict_proba")` and the same with `HistGradientBoostingClassifier(max_iter=100)`; for region (6 classes) use macro one-vs-rest AUC. Bootstrap CI by resampling the predictions (B=200).
2. **Permutation null**: shuffle `A` 100 times (use a 5000-row subsample for speed), recompute the logistic AUC, report the 95th percentile; `p_value` = share of null AUCs at least the real AUC.
3. **Planted-proxy recall**: fit the logistic attacker on all rows, run `sklearn.inspection.permutation_importance`; `planted_top = importance argmax == "night_active_share"` for gender, `"local_market_share"` for region.
4. **Equal-risk check**: bin by `r_true` deciles; within each bin compute the approval-rate gap between levels of `A`; report the mean absolute gap. Large AUC with a small equal-risk gap = "reconstructible is not the same as discriminatory".
Output dict per attribute: `{"attribute", "auc": [lo, est, hi], "auc_boosted": [...], "permutation_null_95": float, "p_value": float, "planted_top": bool, "conditional_gap": [..]}`. Contract: `proxy_leakage` is a list of these.

## Step 6. `audit/nolabel.py`: bounds when group labels are missing (45 min)

Situation: the lender does not have the attribute `A` (say, gender) but knows a proxy cell `z` per person (region) and the external table `P(A=1 | z)`. We observe approval `Y` by cell. The disparity `P(Y=1|A=1) - P(Y=1|A=0)` is not point-identified; it is bounded. This is the partial-identification idea of Kallus, Mao and Zhou (arXiv 1906.00285). **Verify against the paper before presenting as theirs.** The following linear program gives sharp bounds under exactly these constraints and is self-contained:

Variables `a[z,y,g] >= 0` = joint probability of cell z, outcome y, group g. Constraints:
- for each `z, y`: `sum_g a[z,y,g] = P(z, y)` (observed from data);
- for each `z, g`: `sum_y a[z,y,g] = P(z) * P(A=g | z)` (the proxy table).
Then `P(A=g) = sum_z P(z) P(g|z)` is known and `P(Y=1 | A=g) = sum_z a[z,1,g] / P(A=g)`, which is linear in the variables. Minimise and maximise the disparity with `scipy.optimize.linprog` (method `highs`).
```python
def nolabel_bounds(proxy_table, approval_by_cell):
    """proxy_table: columns cell, p_a1 (P(A=1|cell)), share (P(cell)).
       approval_by_cell: columns cell, approval_rate. Returns {"lower","upper","width"} for P(Y=1|A=1)-P(Y=1|A=0)."""
```
Build `c` (objective) and `A_eq`, `b_eq` from the above; one linprog call with `c` and one with `-c`.

UI: always show only the interval, never a point estimate. Test (`test_p2_nolabel.py`): (1) with pure cells (`p_a1` in {0,1}) the interval collapses to the true disparity (width < 1e-6); (2) with uniform cells (`p_a1` equal everywhere) the interval is wide (width > 0.3 for a toy example with approval rates 0.2 and 0.8 across cells); (3) in 100 simulated draws the true disparity lies inside the interval every time. Expected on our data: region is informative for urban_rural but the gender share is about the same in every region, so the gender interval stays wide: "this shows how little a weak proxy tells you".

## Step 7. `audit/tradeoff.py` (20 min)

For each group level: base rate (true default rate), PPV, TPR, and the identity `FPR = (p / (1 - p)) * ((1 - PPV) / PPV) * TPR` where `p` is the prevalence of GOOD (positive = repays, prediction positive = approved). Show the implied FPR next to the measured FPR. Text: "When base rates differ, equal calibration, equal TPR and equal FPR cannot all hold (Kleinberg et al.; Chouldechova). Effort and burden figures are diagnostics, not targets." Output `tradeoff = {"base_rates": {"gender:F": 0.11, ...}, "note": str}`.

## Step 8. `audit/report.py::run_audit` and `scripts/run_audit.py` (45 min)

```python
def run_audit(tau, bootstrap=True):
    df = load(); scorer = load_model()
    # call group_table, burden (all weight sets), failure panel, proxy, tradeoff
    # months_to_approval: json.load(ART_DIR/"recourse_by_group.json")["groups"] if the file exists else None
    out = {"tau": tau, "config_hash": scorer.config_hash, "disclaimer": DISCLAIMER, ...}   # all keys in SPLIT_CONTRACT.md
    json.dump(out, open(C.ART_DIR / "audit_summary.json", "w"), default=float)
    return out
```
`DISCLAIMER = "Synthetic data; an audit-tool demonstration, not a finding about real lenders and not a lending decision tool."` Numpy values must be converted to Python floats before dumping (`default=float` does it). Expected runtime with bootstrap on 20,000 people: a few minutes; cache with `functools.lru_cache` on `(tau, bootstrap)` and use `B=200` while developing.

`scripts/run_audit.py` calls `run_audit(scorer.tau)`, then `run_uncertainty(...)` and `run_poolfresh(...)` (steps 9 and 11) so one command rebuilds all your JSON.

**Null-world test** (`tests/test_p2_null_world.py`): run the audit on `data/synth_null` (Part 1 creates it; skip the test with `pytest.skip` when the folder does not exist, but run it manually at merge). In that world base rates differ by rural/urban but nothing is biased. Assert: all of `failure_flags` are False, burden ratio worst/best below 1.5, `proxy_leakage` planted_top is False for gender (the planted proxy has no effect there). Also a **control run** on `data/synth` generated with `BIAS_FLAGS=False`. Set `null_world_quiet` in the JSON to the result.

---

## Step 9. `uncertainty/exploration.py` (20 min)

The lender only sees outcomes for approved people plus a few explored rejects. Part 1 stores one realisation at `eps = 0.03`; you need other values for the slider, so redraw it:
```python
def redraw(df, eps, seed=0):
    r = np.random.default_rng(seed)
    explored = (~df.approved_legacy.astype(bool)) & (r.random(len(df)) < eps)
    observed = df.approved_legacy.astype(bool) | explored
    prop = np.where(df.approved_legacy == 1, 1.0, eps)       # e(x): known by design
    return explored.values, observed.values, prop
```
`eps = 0` makes `prop = 0` for rejects and breaks the weights: raise `ValueError` for `eps <= 0` unless the caller passes `allow_zero=True` (then return a warning flag; this is the "deterministic rule breaks positivity" demonstration).

## Step 10. `uncertainty/conformal.py` (1.5 h)

Method: weighted Mondrian split conformal. Nonconformity of a labelled example is `s = 1 - p_hat(y | x)` where `p_hat(1|x) = p_default`, `p_hat(0|x) = 1 - p_default`.

```python
import numpy as np, joblib

CELLS = ["thin_file", "gender"]          # Mondrian partition; config

def wquantile(s, w, level, w_test):
    """Smallest s_k such that weight of {s <= s_k} / (sum w + w_test) >= level; inf if never."""
    o = np.argsort(s); s, w = s[o], w[o]
    cum = np.cumsum(w) / (w.sum() + w_test)
    k = np.searchsorted(cum, level)
    return s[k] if k < len(s) else np.inf

class ConformalModel:
    def fit(self, df_cal, weights, alpha, cells=CELLS):
        """df_cal: observed calibration rows (columns p_default, y_obs, + cell columns)."""
        self.alpha, self.cells, self.q, self.ess = alpha, cells, {}, {}
        s = np.where(df_cal.y_obs == 1, 1 - df_cal.p_default, df_cal.p_default)
        for key, idx in df_cal.groupby(cells).indices.items():
            w = weights[idx]
            self.q[key] = wquantile(s[idx], w, 1 - alpha, w.mean())
            self.ess[key] = float(w.sum() ** 2 / (w ** 2).sum())
        self.q_global = wquantile(s, weights, 1 - alpha, weights.mean())
        return self

    def predict_set(self, p_default, groups):
        out = []
        for p, (_, g) in zip(np.atleast_1d(p_default), groups.iterrows()):
            q = self.q.get(tuple(g[c] for c in self.cells), self.q_global)
            sets = [y for y, sc in ((0, p), (1, 1 - p)) if sc <= q]     # y=0 has score p, y=1 has score 1-p
            out.append({"set": sets, "abstain": len(sets) != 1})
        return out
```
Check the nonconformity convention once: for label 0 the score is `1 - (1 - p) = p`; for label 1 it is `1 - p`. That is what the code above uses. Save with `joblib.dump(model, C.ART_DIR / "conformal.joblib")`; `load_conformal()` loads it.

`run_uncertainty(eps, alpha)`:
1. `df = load()`; `cal = df[(df.split == "val")]`; apply `redraw(cal, eps)` → `observed`, `prop`; keep observed rows and set `y_obs` to `y_true` there (the redraw changes who is observed; outcomes come from `y_true`).
2. Three calibration variants, each producing a `ConformalModel`:
   - `unweighted_accepts`: observed rows with `approved_legacy == 1` only, all weights 1;
   - `unweighted_explored`: rows with `explored == 1` only, weights 1 (if fewer than 50 rows, mark the variant `"too few rows"`);
   - `weighted`: all observed rows, `weights = np.minimum(1 / prop, C.W_MAX)`.
3. Evaluate each on `df[df.split == "test"]` (everyone has a known `y_true`, so coverage can be measured on the fresh population): per group level (all 7 attributes) `coverage = mean(y_true in set)`, `referral = mean(abstain)`, `n`, and `ess` of that level's cell(s) (take the minimum over cells containing it). Also `coverage_gap` and `referral_gap` = the maximum over attributes of (max - min across levels with n >= 50). **Show both gaps together:** exact group coverage and equal set size cannot both hold (Gao et al.).
4. Write `ART_DIR/uncertainty.json` per the contract, plus `conformal.joblib` for the `weighted` model at the default `eps`.
5. Warn (a `"warnings"` list in the JSON) when any cell has `ess < 30` or when `eps <= 0.01`.

Do not claim that accepts-only calibration always undercovers; report what you see.

Tests (`test_p2_conformal.py`): (1) with `eps = 0.05` on a toy population, weighted coverage on the fresh test population is within 0.03 of `1 - alpha` (use `N=20000` fixtures or the real data); (2) ESS equals `n` when all weights are 1; (3) `redraw(eps=0)` raises; (4) `predict_set` returns an abstain for a case with `p = 0.5` when `q < 0.5` and a singleton for `p = 0.02`; (5) `wquantile` of `[0.1, 0.2, 0.3]` with equal weights at level 0.5 and `w_test` = 1 equals `0.2`.

## Step 11. `uncertainty/pool_vs_fresh.py` (45 min, cut first if short)

Goal: show that monitoring a lender's own approved pool can look healthy while screening on the real population gets worse.
```python
def run_poolfresh(eps, rounds=20, batch=1500, seed=0):
    # fresh = test split (everyone labelled). stream = train split rows, shuffled, consumed batch by batch.
    # round 0: model_0 = logistic fitted on the first batch's observed rows chosen by the legacy rule.
    # each round r: take the next batch; approve if model_{r-1}.p_default <= tau; add explored rejects with prob eps;
    #               append observed rows (outcome = y_true) to the pool; refit a LogisticRegression on the pool;
    #               pool_recall = share of defaulters in the POOL that the new model would reject (p > tau)
    #               fresh_recall = share of defaulters in FRESH that the new model would reject
    # write ART_DIR/poolfresh.json = {"round": [...], "pool_recall": [...], "fresh_recall": [...]}
```
Use `sklearn.linear_model.LogisticRegression` on standardised `scorer.features`, `tau = scorer.tau`. With `eps = 0`, fresh recall should drift below pool recall (selection bias); with `eps = 0.05` they stay closer. Run for `eps in {0, 0.03, 0.10}` and write all three series under one key per eps. Message on the page: "monitoring on the pool can look healthy while screening on the real population does not". Do not claim more than the plot shows.

## Step 12. Pages (2.5 h total)

Both pages begin with `import sys; sys.path.insert(0, "src")`, load their JSON from `C.ART_DIR` (offline mode, no API), show an error box with the command to regenerate (`python scripts/run_audit.py`) if the file is missing, and end with a caption containing the disclaimer and `config_hash`. All tables show `n`.

**`app/pages/2_Audit.py`**, sections in this order:
1. Banner: "Audit-tool demonstration on synthetic data."
2. Group table (all columns, with the interval shown as `est [lo, hi]`), grey rows where n < 50, an attribute selector.
3. Burden: bar chart per group level with error bars, the worst group highlighted, text "burden among the creditworthy who were rejected". Table of `infeasible_share`.
4. Effort-budget curves (line chart, one line per level of the selected attribute).
5. Risk-matched gap (bar per risk quintile).
6. Level-down: line chart approval rate vs worst-group burden, with a `tau` slider that moves a marker on the curve (from the precomputed `level_down` list, nearest tau); show the warning card when `failure_flags.level_down` is true.
7. Failure-mode cards: level-down, inversion, gaming, infeasible share, referral view.
8. Weight sensitivity: a small table of the worst group and sign under each weight set.
9. Proxy-leakage card: AUC with interval, null, planted-proxy recall, equal-risk gap; message "reconstructible is not the same as discriminatory".
10. No-label mode: dropdown with built-in region proxy table (`P(A=1|region)` from the data), a `st.file_uploader` accepting a CSV with columns `cell,p_a1,share,approval_rate`, calls `nolabel_bounds` directly, shows the interval only.
11. Trade-off box.
12. Months to approval by group from `months_to_approval` (hide the whole section with a note when it is `null`).

**`app/pages/4_Uncertainty.py`**:
1. Sliders: `eps` (0.01 to 0.10, step 0.01, default 0.03) and `alpha` (0.05 to 0.20, default 0.10). Button "Recompute" calls `run_uncertainty(eps, alpha)` directly (takes seconds); otherwise it reads the saved JSON.
2. Side by side: coverage by group and referral by group, with both gaps in large numbers.
3. The three calibration variants as three small tables; ESS column; warnings list.
4. Pool vs fresh line chart.
5. The assumptions list (copy from `docs/IMPLEMENTATION_SPEC.md` 5.2) and the sentence: "Applicants and loan officers may not understand conformal sets; this is an engineering demonstration."

## Step 13. Acceptance for Part 2

- `~/.venv/bin/python -m pytest tests/test_p2_*.py -q` passes on fixtures and on real data.
- `run_audit(scorer.tau)` returns a dict containing every key listed in `docs/SPLIT_CONTRACT.md` and writes `audit_summary.json`.
- Both pages open with no exceptions on fixtures and on real data.
- The null-world and control runs keep every flag False.
- Cut order if you are short on time: pool-vs-fresh, sensitivity of weights, referral view, proxy card. Never cut: group table, burden, level-down, basic conformal.
