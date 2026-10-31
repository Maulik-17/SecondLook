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
Audit highlights (`artefacts/audit_summary.json`, tau 0.20, test split, n = 4016; outcome metrics are among approved applicants, about 39% of each group has no observed outcome):

| Group | n | Approval | Base default rate |
|---|---|---|---|
| Women | 1202 | 0.884 | 0.096 |
| Men | 2814 | 0.882 | 0.112 |
| Thin file | 2096 | 0.869 | 0.111 |
| Not thin file | 1920 | 0.897 | 0.102 |
| Shared phone | 895 | 0.891 | 0.116 |
| Document poor | 1234 | 0.878 | 0.113 |

Approval rates are close across these groups. The burden on creditworthy rejected applicants differs more: the worst-to-best ratio is 2.29 (worst group region R6, best salaried_informal), and thin-file applicants have a higher burden estimate (0.07) than other applicants (0.04). The level-down flag is raised; inversion and gaming flags are quiet. On the null cohort (unequal base rates, no injected bias) the audit stays quiet. These are properties of the injected generator, not findings about real lenders.

## Data
See `docs/DATASHEET.md`.

## Ethical considerations
- Selective labels: outcomes are observed only for approved or explored applicants; the audit corrects with exploration propensities.
- Proxies: two planted proxy features have no effect on true risk by construction; the audit checks whether they can be reconstructed.
- Assumed signals: rent, utility and recharge on-time rates are assumed to predict default, and the current model dropped them.
- Burden and recourse effort are diagnostics. Recourse action costs are assumptions. Explaining how a score reacts to behaviour can enable gaming and model extraction.

## Caveats
Synthetic circularity: the generator injects the biases the audit then looks for, so any bias found is a property of the generator, not of the world. AUC around 0.71 reflects deliberate hidden frailty in the generator. No validation on real borrowers.
