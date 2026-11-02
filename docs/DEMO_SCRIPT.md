# 3-minute demo script

Target 2:50, hard limit 3:00. About 400 spoken words at a calm pace (roughly 140 words a minute). Screen names follow the sidebar; if the redesigned UI renames a page, keep the order and change only the names.

Before recording: run `python scripts/start.py`, wait for the app, open Applicant, Audit and Recourse once so nothing is cold, browser zoom 110%, 1080p, microphone check. Pick the applicants in advance: one approved gig worker, one thin-file referral, one rejected applicant with a plan that comes back in under 5 seconds. Rehearse once with a timer, then record in one or two takes. Record the screen and voice together so the cuts do not drift.

| Time | Screen | Say (short form) |
|---|---|---|
| 0:00 to 0:20 | Home | Problem 3.4: people with no credit history are rejected by default. Scoring them is the easy part. The hard question is who pays the cost of getting approved afterwards. The data are synthetic, calibrated on Czech Berka data. This is an audit-tool demonstration, not a finding about real lenders. |
| 0:20 to 0:45 | Applicant | Consent notice first. Pick a gig worker: score, decision, plain-language reasons. Then a thin file: the model is not sure, so it refers the case to a person instead of guessing. |
| 0:45 to 1:30 | Audit | Group table with intervals and counts. The key number is burden: among creditworthy people who were rejected, which group has to change the most to get approved. Drag tau. The level-down warning appears: a gap closed by rejecting more people is not fairness. |
| 1:30 to 1:55 | Audit (proxy, no-label) | A proxy for gender can be rebuilt from behaviour. When group labels are missing the bounds are wide. A weak proxy tells you little, and we show that instead of hiding it. |
| 1:55 to 2:30 | Recourse | Rejected applicant, press "Find cheapest plan". The answer is a probability, not an instruction: chance of approval within 12 months with an interval, and the fan chart of income futures. Stress table: a plan that looks good can be fragile. Move the persistence slider, press the button again: the probability changes. Costs are assumptions. |
| 2:30 to 2:50 | Uncertainty, Data | Coverage and referral gaps side by side. The data page labels every input as assumed or real. |
| 2:50 to 3:00 | Home | Limits again: synthetic data, assumed signals, no compliance claim. Thank you. |

If the recording runs over, cut in this order: the proxy and no-label segment, then the Uncertainty and Data segment. Never cut the opening disclaimer, the burden number, or the Recourse probability.

## If you are asked

- **Does the audit detect your injected biases?** No, and we say so. The control run over four seeds showed the level-down flag and the unexplained-gap flag fire with the biases on and off, and the burden ratio was not reliably larger with them on. The flags are a sensitive screen and the numbers describe the generator, not the world (`docs/CONTROL_RUN.md`).
- **Null-world check:** quiet only at tau 0.20 or higher; stricter thresholds give false alarms.
- **Why does the "gaming" row equal the base case?** The trained model dropped the three on-time features, so the on-time lever and the gaming stress do nothing. The page says so under the stress table. A variant that makes them informative exists but is off by default (`docs/ON_TIME_VARIANT.md`).
- **Why do many plans show as fragile?** The "retrain" stress row has the lender refit on drifted data with its own stricter threshold.
- **Is this Indian data?** No. Simulated, with parts calibrated on Berka (Czech, 1990s). No validation on Indian thin-file borrowers.
- **Why not a booster?** A monotone XGBoost model was no more accurate than the scorecard (AUC 0.705 vs 0.707) and no clearly more stable (`docs/BOOSTER_COMPARISON.md`).

## Backup if the UI is not ready or fails on stage

Run `python scripts/check_api_live.py`, which starts the API and calls all eight endpoints, or open http://localhost:8000/docs and call `/score` and `/recourse` from there. Say the same story from the API: score with reasons, referral on a thin file, a probability-based plan. `/recourse` can take up to about 5 seconds, so say what it is doing while it runs.
