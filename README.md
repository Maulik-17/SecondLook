# SecondLook: audit-first credit scoring for people with thin credit files

**Status: audit-tool demonstration on synthetic data. Not a lending decision tool. Nothing here is a finding about real lenders.**

SPHINX'26 24-hour hackathon, MNIT Jaipur. Team NexusNavigators.

## 1. Problem statement 3.4

Gig workers, shopkeepers and students often have no credit history and are rejected by default. The task: build a score from rent, utility, recharge and cash-flow signals, check fairness across groups, use synthetic data with a consent-based design, and (bonus) provide a bias audit panel.

<!-- TODO: paste the exact wording of statement 3.4 from the organiser PDF before submission. -->

## 2. What SecondLook adds

A plain cash-flow score is the easy part. SecondLook scores people and then audits the score:

1. **Burden.** Among creditworthy people who were rejected, which group has to change the most to be approved?
2. **Uncertainty.** How sure is the score on thin files? Unsure cases are referred to a person (conformal sets).
3. **Volatility-aware plan.** Which repayment plan still works when income is volatile? We simulate many income futures and report the probability of approval, not a single instruction.
4. **No labels.** What can be said about group gaps when group labels are missing? Bounds, and they are wide when the proxy is weak.

## 3. Screenshots

<!-- TODO: add 2 or 3 PNGs from the running app to docs/img/ and link them here. -->

## 4. Data

- **Simulator with hidden ground truth.** Default depends on a simulated post-loan shortfall (unobserved expense shocks), never on a group label. Income is a persistent, volatile log-income process by occupation.
- **Berka calibration.** Parts of the generator are calibrated on the Berka Czech bank dataset (see `docs/DATASHEET.md`). The data are **not Indian data**.
- **Four injected biases** (switch `BIAS_FLAGS` in `src/secondlook/config.py`, off for a control run): noisier and lower rent/utility/recharge signals for shared-phone or document-poor people; higher income volatility for women in gig work; a planted gender proxy (`night_active_share`); a planted region proxy (`local_market_share`). The planted proxies have no effect on true risk.

| Item | Status |
|---|---|
| Income, balances, volatility, shocks | simulated, calibrated on Berka where noted in the datasheet |
| Default outcome | simulated from hidden shocks |
| Rent, utility, recharge on-time signals | **assumed** to predict default; no paper we read shows it |
| Night-active and local-market shares | **assumed**, planted proxies |
| Recourse action costs and levels | **assumed** (`src/secondlook/recourse/actions.py`) |
| Berka loans | real (Czech, 1990s), used for calibration and a sanity run only |

No caste or religion attribute exists anywhere.

## 5. Architecture

```
Berka (calibration) --> sim/ generator --> data/synth/*.parquet
                                              |
                          model/ scorecard (logistic, calibrated, cost-based tau)
                              |            |             |
                       audit/ (burden,  uncertainty/   recourse/ (AR(1) income,
                       proxy, no-label) (conformal)    Monte Carlo, stress tests)
                              \            |             /
                       Streamlit app (app/) + FastAPI (src/secondlook/api)
```

## 6. How to run

```
git clone <repo> && cd SecondLook
bash scripts/fetch_berka.sh        # optional: downloads the Berka dump
python scripts/make_data.py && python scripts/train.py
python scripts/run_recourse.py && python scripts/run_audit.py
streamlit run app/Home.py
uvicorn secondlook.api.main:app --port 8000
pytest -q
```

Set `PYTHONPATH=src`. Packages: numpy, scipy, pandas, pyarrow, scikit-learn, xgboost, statsmodels, fastapi, uvicorn, streamlit, plotly, pydantic, joblib, pytest; optional shap, fairlearn, mapie. For quick work before real data exists: `python scripts/make_fixtures.py`, then `export SL_DATA=data/fixtures/synth SL_ART=data/fixtures/artefacts` (random numbers in the correct schema).

<!-- TODO: confirm these commands in a fresh clone before submitting. -->

### Recourse in one paragraph

A rejected applicant asks what to change over the next H months. Income is fitted as an AR(1) process on log inflow using only the months the model sees (thin files: 3 to 5), with shrinkage towards an occupation prior. For each of 81 candidate plans (cut spending, extra income, on-time payments, smaller loan) we draw 500 to 1000 income paths with common random numbers, rescore each month with the same model, and record the first month with `p_default <= tau`. We return the cheapest plan whose probability of approval within H months reaches the target (default 0.8), a fan chart, and four stress tests (volatility drift, stricter threshold, removing the assumed on-time signals, income persistence). A plan is flagged fragile if it meets the target but falls below 0.6 under any stress. It is a model-conditional probability on synthetic data.

## 7. Stack

scikit-learn, FastAPI, Streamlit, plotly, XGBoost (optional), SHAP (optional).

## 8. Team

Harshit Jangir (leader), Nikhil Saini, Shreya Chaudhary, Maulik Pratap Singhal.

## 9. Limits and claims we do not make

- Data are synthetic or Berka (Czech), not Indian.
- Rent, utility and recharge signals are assumed. In the current trained model the three on-time features were dropped by feature selection, so the recourse "on-time" action and the "gaming" stress test have no effect on the score (`artefacts/metrics.json`, `dropped_features`).
- No validation on Indian thin-file borrowers.
- Burden and effort are diagnostics, not targets. Recourse costs are assumptions in effort points.
- The four-fifths rule has no legal force in India.
- No compliance claim for RBI FREE-AI or DPDP.
- No claim that borrowers or loan officers understand reason codes or conformal sets.
- Post-processing fairness mitigation is not implemented.
- Recourse reveals how the score reacts to behaviour; real deployments must consider misuse and model extraction.
- Any bias found is a property of the injected generator, not of the world.

## 10. Attribution

scikit-learn, XGBoost, FastAPI, Streamlit, plotly. Berka: P. Berka, PKDD'99 discovery challenge, obtained from the CTU Prague Relational Learning Repository; Motl and Schulte, "The CTU Prague Relational Learning Repository", arXiv:1511.03086. Methods adapted (reimplemented, no code copied): Fang et al. 2607.28847, Barrainkua et al. 2509.04128, De Toni et al. 2410.08007, Plecko 2605.11373, Kallus, Mao and Zhou 1906.00285, Tibshirani et al. (conformal under covariate shift), Scarone and Baeza-Yates 2606.18479.

<!-- TODO: spot-check each arXiv id against the paper title before submitting. -->

## 11. Licence and disclaimer

<!-- TODO: team to confirm licence; MIT proposed. -->
MIT (proposed). **Audit tool demonstration, not findings about real lenders.**
