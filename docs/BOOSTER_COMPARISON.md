# Booster comparison (optional)

Question: does a monotone-constrained XGBoost model do better than the logistic scorecard, and is its SHAP explanation as stable as the scorecard's coefficients?

How to reproduce: `python scripts/train.py`, then `python scripts/run_booster.py`. The numbers are written to `artefacts/booster_comparison.json`. Everything below comes from that file for one simulated dataset and one seed. It says nothing about real applicants.

## Set-up
- Same train, validation, test and out-of-time (oot) splits, and the same 10 kept features, as the logistic scorer.
- Booster: 200 trees, depth 3, learning rate 0.05. Each feature has the same direction constraint as in `contracts.FEATURE_DIRECTION` (the unconstrained features stay unconstrained).
- Platt calibration on the validation split, and the cost-based threshold chosen by the same search as the scorecard (cost 5 per missed default, 1 per lost good customer).

## What the numbers show

| | Logistic test | Booster test | Logistic oot | Booster oot |
|---|---|---|---|---|
| AUC (95% CI) | 0.707 (0.680-0.734) | 0.705 (0.679-0.733) | 0.718 (0.692-0.745) | 0.710 (0.683-0.737) |
| Brier | 0.0891 | 0.0882 | 0.0874 | 0.0868 |
| ECE | 0.0056 | 0.0068 | 0.0082 | 0.0060 |
| Approval rate | 0.882 | 0.829 | 0.893 | 0.835 |

- Ranking quality is the same within noise: the AUC intervals overlap almost completely. The booster has no accuracy advantage here.
- Brier is slightly lower for the booster, by about 0.001. We did not test whether that gap is larger than noise.
- Approval rate differs mostly because the cost search picked a lower threshold for the booster (0.17 versus 0.20). That is a policy difference, not a sign that one model is better.
- Largest group approval-rate gap (test): region, 0.074 for logistic and 0.059 for the booster. Occupation and shared-phone gaps are larger for the booster on test (0.036 versus 0.021, 0.023 versus 0.010); thin-file is smaller (0.017 versus 0.028). On oot the booster's occupation gap is larger (0.064 versus 0.030). No model is uniformly fairer in these numbers. The full per-group rates are in the JSON. The two models use different thresholds, so gaps are not strictly like-for-like.

## Stability across 10 bootstrap refits of the training split
| | Mean rank correlation of importances | Mean share of top-5 features shared between two refits |
|---|---|---|
| Booster, mean abs SHAP (on test) | 0.907 | 0.809 |
| Logistic, abs standardised coefficients | 0.825 | 0.858 |

The two methods are close. The booster's full ranking is a little more stable, while the logistic top-5 is a little more stable. With 10 refits and 10 features we would not call either one clearly more stable.

## Reading
On this synthetic data the booster does not improve accuracy over the scorecard, and it is not clearly better on group gaps or explanation stability. The scorecard is simpler to read and to audit, so it remains the main model. The booster is kept as a cross-check showing that the conclusion does not depend on the model family.

Limits: one dataset, one seed, 10 refits, no significance tests beyond the AUC intervals, default hyperparameters not tuned.
