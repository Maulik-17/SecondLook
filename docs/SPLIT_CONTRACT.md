# Three-person split and integration contract

Rule: the parts talk ONLY through (a) parquet/JSON files in `config.DATA_DIR` / `config.ART_DIR` and (b) the function signatures below. Nobody imports another part's internals. Shared code that already exists and is frozen: `src/secondlook/config.py`, `contracts.py`, `model/scorer.py` (interface), `scripts/make_fixtures.py`.

**Start now, before Part 1 delivers:**
```
~/.venv/bin/python scripts/make_fixtures.py
export SL_DATA=data/fixtures/synth SL_ART=data/fixtures/artefacts
```
This writes random data in the exact schema plus a working model. Unset both variables when Part 1 publishes the real files. Fixture numbers mean nothing; only the schema is real.

## Who builds what

| Part | Owns (spec sections) | Source dirs | UI pages | Est. |
|---|---|---|---|---|
| **1 Data, model, API** | 2 Data, 3 Model, 6 explain, 8 API, `scripts/*`, tests for sim/scorer | `berka/ sim/ model/ api/` | Home, Applicant, Data_and_Limits | ~11 h |
| **2 Audit and uncertainty** | 4 Audit, 5 Uncertainty | `audit/ uncertainty/` | Audit, Uncertainty | ~10 h |
| **3 Recourse, glue, submission** | 7 Recourse, README, model card, datasheet, video, demo script | `recourse/` | Recourse | ~9 h + docs |

Parts 2 and 3 read Part 1's files and never write them. Part 2 and 3 are linked only by `recourse_by_group.json` (below). Cut order if behind is in spec section 10.

## Files Part 1 publishes (parquet, one row per person unless stated)

`id` is an int, the same in every table. `y` = 1 means default. Required columns are enforced by `contracts.check(name, df)`; call it before writing.

| File (`DATA_DIR/`) | Columns |
|---|---|
| `people.parquet` | `id, split{train,val,test,oot}, gender{F,M}, urban_rural{urban,rural}, region{R1..R6}, occupation{gig_driver,gig_delivery,shopkeeper,student_parttime,salaried_informal}, thin_file{0,1}, shared_phone{0,1}, document_poor{0,1}, loan_amount, loan_term, y_true, r_true` (r_true = the generator's true default probability) |
| `features.parquet` | `id` + the 17 names in `contracts.FEATURES` (no group, no latent) |
| `monthly.parquet` (many rows per id) | `id, month, inflow, outflow, balance` (month 1..24 history, 25..36 outcome window) |
| `scores.parquet` | `id, p_default, logit, decision{1=approve}, tau` for every person and every split |
| `lender_history.parquet` | `id, approved_legacy, explored, propensity, observed, y_obs` (propensity = e(x) = probability of having an observed outcome; y_obs is NaN when not observed) |
| `latents.parquet` | Part 1 and recourse ground truth only: `contracts.LATENTS_COLUMNS` |

| Artefact (`ART_DIR/`) | Content |
|---|---|
| `model.joblib` | a `Scorer` (below) |
| `data_dictionary.json` | `[{"feature": str, "source": "berka-supported"\|"literature-supported"\|"assumed", "description": str, "direction": -1\|0\|1}]` |
| `berka_sanity.json`, `oot.json` | metrics with CIs for the Data_and_Limits page (`{"auc": [lo, est, hi], ...}`) |

### Scorer interface (frozen; internals are Part 1's)
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
`model/scorer.py` is a working reference version; Part 1 may replace the internals but must keep this interface. Part 1 also provides `secondlook.model.features.cash_features(inc, out, bal, emi) -> dict[str, ndarray]` where `inc, out, bal` are arrays of shape `(K, w)` (K paths, w months, oldest first) and `emi` has shape `(K,)`; it returns the 12 cash-derived features (`mean_inflow, median_inflow, inflow_cv, dip_months, spike_months, weeks_of_buffer, low_balance_days, expense_to_income, balance_level, balance_slope, months_observed, emi_to_income`). The simulator and Part 3 both call it, so features can never drift apart. The 5 non-cash features (three `*_on_time`, two planted proxies) are held at the person's stored values by Part 3. Another tau than `s.tau` is passed by the caller as an argument; nothing mutates `s`.

## Part 2 deliverables

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

## Part 3 deliverables

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
`None` means "never within the horizon". Part 3 also writes `README.md`, `docs/MODEL_CARD.md`, `docs/DATASHEET.md`, the demo script and the video, and owns the merge checklist below.

## API (Part 1 wires it; handlers call the functions above)

Endpoint table is spec section 8. Mapping: `/score` → `Scorer` + `load_conformal().predict_set` + `explain.reasons`; `/recourse` → `plan()`; `/audit/summary` → cached `audit_summary.json`; `/uncertainty/*` → the JSON files; `/data/dictionary` → `data_dictionary.json`. Until Part 2 or 3 deliver, return HTTP 501 `{"detail": "not ready"}` on those routes; do not mock them.

`/score` reasons shape: `[{"feature": str, "contribution": float, "direction": "raises risk"|"lowers risk", "text": str}]`, top 5, never mentioning group attributes.

## UI rule

Each page loads the files/functions directly (offline mode) and does NOT need the API running. Charts: plotly. Every table shows n. Every page shows `disclaimer` and `config_hash`. Page file names are fixed: `app/Home.py`, `app/pages/1_Applicant.py` (P1), `2_Audit.py` (P2), `3_Recourse.py` (P3), `4_Uncertainty.py` (P2), `5_Data_and_Limits.py` (P1). `app/Home.py` is Part 1's.

## Git and merge

- Branches: `part1-data-model`, `part2-audit`, `part3-recourse`. Touch only your own directories. Changes to `config.py`, `contracts.py` or the Scorer interface need a message to the other two first.
- Commit tests with your code. Never mention AI tools in commit messages.
- Raw Berka, `data/synth`, `data/fixtures`, `artefacts/` are never committed.

## Merge checklist (in this order)

1. Part 1 pushes the real generator, `make_data.py` and `train.py`; the others unset `SL_DATA SL_ART`, run both scripts.
2. Run `pytest` on all three branches merged; every `contracts.check` passes.
3. Part 3 runs `run_recourse_by_group`; Part 2 then runs `run_audit` and `run_uncertainty` (it picks up months-to-approval).
4. Start the app (`streamlit run app/Home.py`) and open every page; start the API and call every endpoint once.
5. Null-world test and bias-flags-off control: the audit panel must stay quiet.
6. Freeze 8:30 AM; Part 3 finishes the README; video 9:45 to 11:15 AM.

## Open risks

- The real simulator may not reach the calibration targets in time. Fallback: ship the simpler generator with the documented hidden-truth mechanism and say so in the datasheet.
- Part 3 carries the most (recourse plus submission material). If behind, cut recourse stress tests first (spec cut order) and move the demo script to whoever finishes first.
