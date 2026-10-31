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

- In the main seed, with the biases off, the flags, the unexplained gap and the planted-proxy detection all went quiet, and the burden ratio fell from 2.29 to 1.56. **This did not hold up across seeds (see the table below), so do not read it as a general result.**
- A burden ratio above 1 remains with no injected bias. It reflects real differences in income volatility and base rates between groups in the generator, so a ratio above 1 is not by itself evidence of bias.
- The null-world check run on the control scorer at its own threshold (0.15) was **not** quiet: the level-down flag and the unexplained-gap flag both fired on the null cohort. We traced this to the threshold, not the scorer. Running the null cohort at fixed thresholds gives the same result for both scorers: at tau 0.10 and 0.15 the unexplained-gap flag fires (and level-down at 0.15 for both scorers, and at 0.10 for the main scorer), and at tau 0.20 and 0.25 everything is quiet. So the audit is quiet on the null world at tau 0.20 or higher and raises false alarms at stricter thresholds, where the unequal base rates of that cohort show up as a gap. Treat the level-down and unexplained-gap flags as sensitive screens, not proof of harm, and read them together with the threshold in use. We did not retune the check.

## Three more seeds (`bash scripts/run_seeds.sh`)

Each seed regenerates the whole world, retrains the scorer and runs the audit in `--fast` mode (no bootstrap), with the biases on and off. These runs use different seeds from the main run, so their numbers are not comparable with the table above.

| Seed | Biases | tau | Test AUC | Worst/best burden ratio | Failure flags raised | Unexplained gap | Null-world quiet |
|---|---|---|---|---|---|---|---|
| 101 | on | 0.18 | 0.708 | 2.10 | 1 | yes | no |
| 101 | off | 0.20 | 0.696 | 1.82 | 1 | yes | no |
| 202 | on | 0.18 | 0.712 | 1.34 | 1 | yes | yes |
| 202 | off | 0.16 | 0.705 | 1.51 | 1 | yes | no |
| 303 | on | 0.17 | 0.712 | 2.04 | 1 | yes | no |
| 303 | off | 0.18 | 0.722 | 2.04 | 1 | yes | no |

What this shows:

- The burden ratio with the biases on (1.34 to 2.10 here, 2.29 in the main seed) is only sometimes larger than with them off. In seed 202 it is smaller; in seed 303 it is identical. With the biases off the ratio is 1.5 to 2.0.
- One failure flag (level-down) is raised in every run, with and without the injected biases, and the unexplained-gap flag is raised in every run too. In these seeds the audit does **not** separate the injected-bias worlds from the bias-off worlds by the burden ratio or the flags.
- So the clean contrast in the main seed (flags quiet when biases are off) was a single-seed result. We do not claim the audit detects the injected biases through these summary numbers. The planted-proxy leakage test is a separate check; it was not compared across these seeds.
- The null-world check is not quiet in 5 of 6 runs, consistent with the threshold dependence above (tau is 0.16 to 0.20 here).
- Caveats: `--fast` audit, three seeds, one run each, and the same default sample size. We have not looked into why the flags fire without injected bias; the likely reason is that the generator already has real group differences in income volatility and base rates, which the flags respond to.
