# Variant: informative on-time signals

In the default data the rent, utility and recharge on-time signals are noisy functions of the account balance, so they add nothing the cash-flow features do not already carry. The sign-constrained scorer drops them, and the on-time recourse lever and the gaming stress test then have no effect.

`SL_ON_TIME_LOAD` (default 0, off) changes the simulator: the on-time rates also fall with the hidden risk, by this many units per standard deviation of the true risk logit. This is an **assumption** (payment discipline reflects risk that cash flow does not show). It is the same assumption the README already labels as unsupported by any paper we read; the variant just builds it into the data. No random draws change, so with the value 0 the data are identical to the default.

Run it into separate folders (nothing in `data/synth` or `artefacts/` is touched):

```
export PYTHONPATH=src SL_ON_TIME_LOAD=0.07 SL_DATA=$PWD/data/ontime/synth SL_ART=$PWD/data/ontime/artefacts
python scripts/make_data.py && python scripts/train.py && python scripts/run_recourse.py && python scripts/run_audit.py
```

Result with 0.07, one seed (31 Oct 2026):

| | Default (load 0) | Variant (load 0.07) |
|---|---|---|
| Test AUC [95% interval] | 0.707 [0.680, 0.734] | 0.721 [0.695, 0.748] |
| Out-of-time AUC | 0.718 | 0.726 |
| Cost-based threshold tau | 0.20 | 0.16 |
| Features dropped by sign constraint | 7, including all three on-time signals | 5, none of the on-time signals |
| Failure flags | level-down raised | level-down raised |
| Worst-to-best burden ratio | 2.29 | 1.78 |
| Null-world check | quiet | **not quiet** |
| `test_real_null_world_folder` | passes | **fails** |

Why it is off by default: with the lower threshold (0.16), the audit's null-world check raises its level-down and unexplained-gap flags on the cohort with unequal base rates and no discrimination. This is the same threshold dependence described in `docs/CONTROL_RUN.md` (the check is quiet at tau 0.20 or higher). We chose to keep the claim "the audit stays quiet on the null world" for the default data, and not to loosen the test. In the variant the on-time recourse lever and the gaming test do act on the score, but we have not re-checked the other tests beyond the first failure (the run stopped there: 46 passed, 1 failed), nor re-taken the screenshots.
