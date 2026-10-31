# Control run: injected biases switched off

Purpose: check that the audit flags what we injected and does not invent findings when the biases are absent.

Method: the same pipeline (`make_data`, `train`, `run_audit`) with `SL_BIAS=0`, which turns off the four injected biases (`scripts/run_control.sh`). Outputs go to `data/control/`, so the main artefacts are untouched. Same seed, same sample size, scorer retrained on the control data.

| | Biases on (main) | Biases off (control) |
|---|---|---|
| Test AUC [95% interval] | 0.707 [0.680, 0.734] | 0.714 [0.688, 0.738] |
| Cost-based threshold tau | 0.20 | 0.15 |
| Failure flags (level-down / inversion / gaming) | level-down raised | none raised |
| Worst-to-best burden ratio | 2.29 (region R6 vs salaried_informal) | 1.56 (region R1 vs region R5) |
| Unexplained gap after matching on risk | yes | no |
| Planted proxies ranked top by the leakage test (gender, region) | yes, both | no |
| Four-fifths ratio, region | 0.92 | 0.94 |

What this shows, and what it does not:

- With the biases off, the flags, the unexplained gap and the planted-proxy detection all go quiet, and the burden ratio falls from 2.29 to 1.56. So most of the main-run signal comes from the injected biases.
- A burden ratio of 1.56 remains with no injected bias. It reflects real differences in income volatility and base rates between groups in the generator, so a ratio above 1 is not by itself evidence of bias.
- The null-world check run on the control scorer at its own threshold (0.15) was **not** quiet: the level-down flag and the unexplained-gap flag both fired on the null cohort. We traced this to the threshold, not the scorer. Running the null cohort at fixed thresholds gives the same result for both scorers: at tau 0.10 and 0.15 the unexplained-gap flag fires (and level-down at 0.15 for both scorers, and at 0.10 for the main scorer), and at tau 0.20 and 0.25 everything is quiet. So the audit is quiet on the null world at tau 0.20 or higher and raises false alarms at stricter thresholds, where the unequal base rates of that cohort show up as a gap. Treat the level-down and unexplained-gap flags as sensitive screens, not proof of harm, and read them together with the threshold in use. We did not retune the check.
- One seed, one run. No repeated-seed variability is reported.
