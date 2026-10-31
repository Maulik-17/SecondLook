# Datasheet: SecondLook synthetic cohort

Audit-tool demonstration data. Synthetic; calibrated on Berka (Czech) where stated. Not Indian data.

## Motivation
Created to demonstrate audit methods for credit scoring of thin-file applicants where ground truth is known to the builders, which real lender data cannot offer.

## Composition
Parquet tables (schema in `docs/SPLIT_CONTRACT.md`, checked by `contracts.check`): `people` (id, split, gender, urban_rural, region, occupation, thin_file, shared_phone, document_poor, loan_amount, loan_term, y_true, r_true), `features` (id + 17 features), `monthly` (inflow, outflow, balance; months 1..24 history, 25..36 outcome), `scores`, `lender_history` (legacy approval, exploration, propensity, observed outcome), `latents` (generator ground truth, for tests only).
Sizes: 20,000 people by default (`N_PEOPLE`); splits train 0.6, val 0.2, test 0.2, plus a separate drifted out-of-time cohort (`oot`). Thin-file people expose only 3 to 5 months of history to the model.

## Collection process
Each person gets an occupation, region and a persistent log-income process with occupation-specific level, volatility and monthly dip probability. Monthly outflows follow income plus fixed costs, and balances accumulate. After the loan, unobserved expense shocks hit with a monthly probability; default arises from instalment shortfall plus unobserved frailty. A legacy lender approves by its own rule and explores a small share at random, which determines which outcomes are observed.

## Injected biases (on when `BIAS_FLAGS = True`)
1. Rent, utility and recharge signals are noisier and lower for shared-phone or document-poor people (shift -0.08, extra noise sd 0.1).
2. Income volatility is 1.25 times higher for women in gig work (drivers, delivery).
3. `night_active_share` carries a gender signal (+0.055 for women): planted proxy, no effect on true risk.
4. `local_market_share` carries a region signal (+0.1 per region index): planted proxy, no effect on true risk.
A control run with the flag off, and a null cohort with unequal base rates and no bias, are available for checking that the audit stays quiet.

## Assumed, calibrated, real
- Assumed: occupation income levels and volatilities, shock probabilities, rent/utility/recharge relation to cash, proxy effects, recourse action costs, decision costs.
- Compared with Berka, not fitted to it: `artefacts/berka_calibration.json` lists the 10/25/50/75/90th percentiles of five features in Berka and in the simulator. Median values (Berka vs simulator): `inflow_cv` 0.22 vs 0.24, `dip_months` 0 vs 1, `weeks_of_buffer` 6.8 vs 22.7, `expense_to_income` 0.99 vs 0.80, `emi_to_income` 0.15 vs 0.11. Only `inflow_cv` is close. The simulator's `weeks_of_buffer` runs about 3 times Berka's, so it is not calibrated to Berka on that feature.
- Real: Berka (P. Berka, PKDD'99) is a real Czech bank dataset with 682 loans, 76 not successful; used for calibration and a sanity run only.

## Preprocessing
Features come from `secondlook.model.features.cash_features` over the visible window, plus three on-time rates and two proxies. For Berka, only data before the loan date is used (temporal rule). Berka features use a 6-month window of 30-day buckets before the loan date, with at least 3 months of history required (`src/secondlook/berka/features.py`); the real-data sanity run uses 5-fold out-of-fold predictions (AUC 0.83, 95% CI 0.77 to 0.88, 76 defaults, so the interval is wide).

## Uses
Demonstrating and testing audit methods. Not for estimating real-world default or bias, not for lending.

## Distribution
Generator code only. Berka is not redistributed; use `scripts/fetch_berka.sh`.

## Maintenance
None after the hackathon.
