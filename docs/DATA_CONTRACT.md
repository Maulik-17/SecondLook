# Data and interface contract

The pipeline stages (data and model, audit and uncertainty, recourse) talk ONLY through (a) parquet/JSON files in `config.DATA_DIR` / `config.ART_DIR` and (b) the function signatures below. A stage does not import another stage's internals. Stable interfaces: `src/secondlook/config.py`, `contracts.py`, `model/scorer.py` (interface), `scripts/make_fixtures.py`.

**Quick start before the real data exists:**
```
python scripts/make_fixtures.py
export SL_DATA=data/fixtures/synth SL_ART=data/fixtures/artefacts
```
This writes random data in the exact schema plus a working model. Unset both variables to use the real files. Fixture numbers mean nothing; only the schema is real.

## Files the data and model stage publishes (parquet, one row per person unless stated)

`id` is an int, the same in every table. `y` = 1 means default. Required columns are enforced by `contracts.check(name, df)`; call it before writing.

| File (`DATA_DIR/`) | Columns |
|---|---|
| `people.parquet` | `id, split{train,val,test,oot}, gender{F,M}, urban_rural{urban,rural}, region{R1..R6}, occupation{gig_driver,gig_delivery,shopkeeper,student_parttime,salaried_informal}, thin_file{0,1}, shared_phone{0,1}, document_poor{0,1}, loan_amount, loan_term, y_true, r_true` (r_true = the generator's true default probability) |
| `features.parquet` | `id` + the 17 names in `contracts.FEATURES` (no group, no latent) |
| `monthly.parquet` (many rows per id) | `id, month, inflow, outflow, balance` (month 1..24 history, 25..36 outcome window) |
| `scores.parquet` | `id, p_default, logit, decision{1=approve}, tau` for every person and every split |
| `lender_history.parquet` | `id, approved_legacy, explored, propensity, observed, y_obs` (propensity = e(x) = probability of having an observed outcome; y_obs is NaN when not observed) |
| `latents.parquet` | generator ground truth, used by recourse and tests only: `contracts.LATENTS_COLUMNS` |

| Artefact (`ART_DIR/`) | Content |
|---|---|
| `model.joblib` | a `Scorer` (below) |
| `data_dictionary.json` | `[{"feature": str, "source": "berka-supported"\|"literature-supported"\|"assumed", "description": str, "direction": -1\|0\|1}]` |
| `berka_sanity.json`, `oot.json` | metrics with CIs for the Data_and_Limits page (`{"auc": [lo, est, hi], ...}`) |

### Scorer interface (stable interface)
```python
from secondlook.model.scorer import load_model
s = load_model()                          # reads ART_DIR/model.joblib
s.predict_proba(features_df) -> ndarray   # calibrated p_default
s.logit(features_df)         -> ndarray   # raw linear score
s.decide(features_df)        -> ndarray   # 1 = approve (p_default <= s.tau)
s.tau: float                              # cost-based default
s.linear_params() -> {"features": [...], "w": ndarray (raw units), "b": float}   # logit = w @ x + b
s.logit_threshold() -> float              # raw logit that corresponds to s.tau
```
`model/scorer.py` is a working reference version; The internals may be replaced but this interface must be kept. The model package also provides `secondlook.model.features.cash_features(inc, out, bal, emi) -> dict[str, ndarray]` where `inc, out, bal` are arrays of shape `(K, w)` (K paths, w months, oldest first) and `emi` has shape `(K,)`; it returns the 12 cash-derived features (`mean_inflow, median_inflow, inflow_cv, dip_months, spike_months, weeks_of_buffer, low_balance_days, expense_to_income, balance_level, balance_slope, months_observed, emi_to_income`). The simulator and the recourse stage both call it, so features can never drift apart. The 5 non-cash features (three `*_on_time`, two planted proxies) are held at the person's stored values by recourse. Another tau than `s.tau` is passed by the caller as an argument; nothing mutates `s`.

## Audit and uncertainty stage

Reads: `people, features, scores, lender_history`, `load_model()`.
```python
# audit/report.py
run_audit(tau: float, bootstrap: bool = True) -> dict          # also writes ART_DIR/audit_summary.json
# uncertainty/conformal.py
class ConformalModel:
    def predict_set(self, p_default: ndarray, groups: DataFrame) -> list[dict]  # [{"set": subset of [0,1], "abstain": bool}]
load_conformal() -> ConformalModel                             # reads ART_DIR/conformal.joblib
run_uncertainty(eps: float, alpha: float) -> dict              # writes ART_DIR/uncertainty.json
# uncertainty/pool_vs_fresh.py
run_poolfresh(eps: float, rounds: int) -> dict                 # writes ART_DIR/poolfresh.json
# audit/nolabel.py
nolabel_bounds(proxy_table: DataFrame, approval_by_cell: DataFrame) -> dict   # {"lower": float, "upper": float, "width": float}
```
**JSON shape** (numbers are `[lo, est, hi]` where marked `[..]`; every row carries `n`):
```
audit_summary.json = {
  "tau": float, "config_hash": str, "disclaimer": str,
  "groups": [{"attribute": "gender", "level": "F", "n": int, "approval": [..], "base_rate": [..],
              "tpr": [..], "fpr": [..], "ppv": [..], "ece": float, "brier": float,
              "missing_outcome_share": float}],            # outcome metrics are "among approved"
  "burden": [{"attribute": str, "level": str, "n": int, "burden": [..], "infeasible_share": float}],
  "burden_headline": {"worst": {"attribute": str, "level": str}, "ratio": float},
  "budget_curve": [{"attribute": str, "level": str, "budget": [float...], "share": [float...]}],
  "risk_matched_gap": [{"attribute": str, "bin": int, "gap": [..]}],
  "level_down": [{"tau": float, "approval_rate": float, "worst_group_burden": float}],
  "failure_flags": {"level_down": bool, "inversion": bool, "gaming": bool},
  "null_world_quiet": bool | null,
  "proxy_leakage": [{"attribute": str, "auc": [..], "auc_boosted": [..], "permutation_null_95": float, "p_value": float,
                     "planted_top": bool, "conditional_gap": [..]}],
  "tradeoff": {"base_rates": {"<attribute>:<level>": float}, "note": str},
  "months_to_approval": [{"attribute": str, "level": str, "n": int, "median_months": float|null, "never_share": float}] | null }
uncertainty.json = {"eps": float, "alpha": float,
  "variants": {"unweighted_accepts": V, "unweighted_explored": V, "weighted": V}}
  with V = {"groups": [{"attribute": str, "level": str, "n": int, "coverage": float, "referral": float, "ess": float}],
            "coverage_gap": float, "referral_gap": float}
poolfresh.json = {"round": [int...], "pool_recall": [float...], "fresh_recall": [float...]}
```
`months_to_approval` is copied from `ART_DIR/recourse_by_group.json` if it exists, else `null` (the page hides the panel).

## Recourse stage

Reads: `people, features, monthly`, `latents` (real run only), `load_model()`.
```python
# recourse/plan.py
plan(person_id: int, horizon: int = 12, target_prob: float = 0.8, rho_override: float | None = None,
     tau: float | None = None) -> dict
# result (also what POST /recourse returns):
# {"person_id": int, "plan": [{"action": str, "amount": float, "cost": float}], "cost": float,
#  "p_success": [lo, est, hi], "months_to_approval_quantiles": {"p10": float|None, "p50": float|None, "p90": float|None},
#  "fan_chart": {"month": [int], "q10": [float], "q50": [float], "q90": [float]},   # projected p_default
#  "stress": {"drift": float, "threshold_shift": float, "gaming": float, "persistence": float},  # P_success under each
#  "fragile": bool}
# recourse/stress.py
run_recourse_by_group(n_per_group: int = 200) -> dict     # writes ART_DIR/recourse_by_group.json
#   = {"groups": [{"attribute": str, "level": str, "n": int, "median_months": float|None, "never_share": float}]}
```
`None` means "never within the horizon". 

## API (handlers call the functions above)

Endpoints are listed at `/docs` when the API runs. Mapping: `/score` → `Scorer` + `load_conformal().predict_set` + `explain.reasons`; `/recourse` → `plan()`; `/audit/summary` → cached `audit_summary.json`; `/uncertainty/*` → the JSON files; `/data/dictionary` → `data_dictionary.json`. Until the audit or recourse files exist, return HTTP 501 `{"detail": "not ready"}` on those routes; do not mock them.

`/score` reasons shape: `[{"feature": str, "contribution": float, "direction": "raises risk"|"lowers risk", "text": str}]`, top 5, never mentioning group attributes.

## UI rule

Each page loads the files/functions directly (offline mode) and does NOT need the API running. Charts: plotly. Every table shows n. Every page shows `disclaimer` and `config_hash`. Pages: `app/Home.py`, `app/pages/1_Applicant.py`, `2_Audit.py`, `3_Recourse.py`, `4_Uncertainty.py`, `5_Data_and_Limits.py`.
