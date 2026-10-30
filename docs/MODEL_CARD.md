# Model card: SecondLook scorecard

Audit-tool demonstration on synthetic data. Not a lending decision tool.

## Model details
- Logistic scorecard on cash-flow features with expert sign constraints, Platt calibration, and a cost-based approval threshold tau (cost of approving a defaulter 5, cost of rejecting a repayer 1; both assumptions, `src/secondlook/config.py`).
- Version: `config_hash` fe1f3c3fd6 (from `artefacts/metrics.json` at the time of writing; refresh if the model is retrained). tau = 0.2.
- Features kept after selection: median_inflow, spike_months, low_balance_days, expense_to_income, balance_level, balance_slope, months_observed, emi_to_income, night_active_share, local_market_share. Dropped: weeks_of_buffer, dip_months, mean_inflow, utility_on_time, recharge_on_time, rent_on_time, inflow_cv.
- No group attribute is an input.

## Intended use
Research and demonstration of audit methods: burden of recourse, uncertainty and referral, volatility-aware recourse planning, no-label bounds.

## Out-of-scope use
Real lending decisions. Any claim about real lenders or about Indian borrowers.

## Factors
Groups audited: gender, urban_rural, region, occupation, thin_file, shared_phone, document_poor.

## Metrics (from `artefacts/metrics.json`, held-out test split, 95% interval)
| Metric | Value |
|---|---|
| Test AUC (n = 4016) | 0.707 [0.680, 0.734] |
| Out-of-time AUC (n = 4000) | 0.718 [0.692, 0.745] |
| Test ECE | 0.0056 |
| Test Brier | 0.089 |
| Test base default rate | 0.107 |
| Approval rate at tau (validation) | 0.887 |

Calibration by gender (test): F n = 1202, mean predicted 0.106, observed 0.096; M n = 2814, mean predicted 0.105, observed 0.112.

Recourse (Part 3): see `artefacts/recourse_by_group.json` for months to approval by group under a fixed moderate plan.
<!-- TODO: add group-table highlights from audit_summary.json once Part 2 has run on the final model. -->

## Data
See `docs/DATASHEET.md`.

## Ethical considerations
- Selective labels: outcomes are observed only for approved or explored applicants; the audit corrects with exploration propensities.
- Proxies: two planted proxy features have no effect on true risk by construction; the audit checks whether they can be reconstructed.
- Assumed signals: rent, utility and recharge on-time rates are assumed to predict default, and the current model dropped them.
- Burden and recourse effort are diagnostics. Recourse action costs are assumptions. Explaining how a score reacts to behaviour can enable gaming and model extraction.

## Caveats
Synthetic circularity: the generator injects the biases the audit then looks for, so any bias found is a property of the generator, not of the world. AUC around 0.71 reflects deliberate hidden frailty in the generator. No validation on real borrowers.
