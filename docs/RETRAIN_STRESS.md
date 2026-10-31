# Retraining stress test (recourse)

A recourse plan takes months. If the lender retrains its scorer meanwhile, the plan may no longer reach approval
(De Toni et al., arXiv 2410.08007, "Time Can Invalidate Algorithmic Recourse"). The `retrain` key in
`stress_tests()` / the `/recourse` response estimates how often that happens.

## Method

1. Once per process (`recourse/retrain.py`, `context.get_retrained()`), build 5 retrained `Scorer`s (about 2 s).
   Each one is fitted on a bootstrap resample of train+val rows in which one third of the rows come from the
   drifted `oot` cohort (volatility x1.2, shocks x1.3, labels included), Platt-calibrated on a bootstrap of
   validation plus `oot` rows, and given its own cost-based threshold `tau` (`set_tau_cost`). Seeds are fixed.
2. For the chosen plan, rerun the same Monte Carlo (K=500, same common random numbers) with each retrained scorer
   judged at its own `tau`, and report the mean over the 5 scorers of P(approved within the horizon).
3. `fragile` = meets the target in the base case and any stress value (now including `retrain`) is below 0.6.

## Limits

- The retrained model replaces the current one from month 1 (conservative); the retrain date is not randomised.
- Only 5 refits, so the number reflects model-to-model spread poorly; it is a mean, not a worst case.
- The retrained scorer is the plain logistic reference `Scorer`, not the sign-constrained production scorecard,
  and it is trained on synthetic labels that include the drifted cohort. Its threshold moves because the cost
  rule is re-applied, not because a lender policy says so.
- It does not model the lender changing features or the applicant reacting to the new model.
- Innovation sd: the earlier recourse version uses innovation sd = sigma, which differs from the guide. That
  approach was kept unchanged; the `drift` test scales it by 1.2.
- Synthetic data only; a model-based probability, not a guarantee.
