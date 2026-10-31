# 3-minute demo script

Under 3:00. Run the app on real (shared) data first, browser zoom 110%, 1080p, microphone on. One rehearsal with a timer, then record in one or two takes.

| Time | Screen | Say |
|---|---|---|
| 0:00 to 0:20 | Home | Problem statement 3.4: people with no credit history get rejected. A plain score is the easy part. The hard question is who carries the cost of fixing a rejection. All data are synthetic or Czech Berka. This is an audit-tool demonstration, not findings about real lenders. |
| 0:20 to 0:50 | Applicant | Consent screen, pick a gig worker, show the score and plain-language reasons. For a thin file: "we are not sure, referred to a person". |
| 0:50 to 1:35 | Audit | Group table with intervals and n. Burden among creditworthy rejected people: worst group highlighted. Drag tau: the level-down warning, "fairer by rejecting more is not fairer". Risk-matched gap. |
| 1:35 to 2:05 | Audit (proxy, no-label) | The proxy can be reconstructed, but the equal-risk gap is small. No-label bounds are wide: a weak proxy tells you little. |
| 2:05 to 2:35 | Recourse | Pick a rejected applicant. Press "Find cheapest plan": plan table, probability of approval within 12 months with interval, months-to-approval quantiles. Point at the fan chart and the tau line. Show the stress table and the fragile badge if present. Move the persistence slider and press the button: the probability changes. Say: costs are assumptions; this is a model-based probability, not a guarantee. |
| 2:35 to 2:55 | Uncertainty, Data | Coverage and referral gaps shown together; exploration slider; the data page labels assumed versus real. |
| 2:55 to 3:00 | Home | Limits stated: synthetic data, assumed signals, no compliance claim. Thank you. |

Notes for the Recourse segment: the current model dropped the three on-time features, so the "gaming" stress row equals the base case (the page says so under the stress table); say so if asked. The stress table also has a "retrain" row: the lender refits the model on drifted data with its own stricter threshold, which is what makes many plans show as fragile. If time allows, one sentence on the control run: with the injected biases switched off, the flags go quiet and the burden ratio falls from 2.29 to 1.56 (`docs/CONTROL_RUN.md`). Caveat if asked: the audit's null-world check is quiet at tau 0.20 or higher and gives false alarms at stricter thresholds.
