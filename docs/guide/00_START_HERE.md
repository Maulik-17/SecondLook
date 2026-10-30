# SecondLook build guide: start here

Read this file first (10 minutes), then ONLY your own part file:

| You are | Read | Branch |
|---|---|---|
| Part 1: data, model, API | `PART1_DATA_MODEL_API.md` | `part1-data-model` |
| Part 2: audit and uncertainty | `PART2_AUDIT_UNCERTAINTY.md` | `part2-audit` |
| Part 3: recourse and submission | `PART3_RECOURSE_SUBMISSION.md` | `part3-recourse` |

Each part file is written as numbered steps. Do the steps in order. Every step says what file to create, what it must contain, and how to check it works. Where the guide gives code, copy it, then adapt only where the step says so.

## 1. What we are building (30 seconds)

SecondLook scores people with thin credit files (gig workers, shopkeepers, students) from cash-flow signals, then audits the score:
1. Who carries the burden of fixing a rejection? (Part 2)
2. How sure is the score on thin files? Refer unsure cases to a person. (Part 2)
3. Which repayment plan still works when income is volatile? (Part 3)
4. What if group labels are missing? (Part 2)

Problem statement 3.4 suggests scikit-learn, SHAP, FastAPI, Streamlit; we use scikit-learn, FastAPI, Streamlit, and optional SHAP.

## 2. Rules that apply to everyone

1. **Data is synthetic or calibrated on Berka (a Czech bank dataset).** Never say it is Indian data. Never say a result is a finding about real lenders. Always say "audit tool demonstration".
2. **Rent, utility and recharge signals are ASSUMED** to predict default. No paper we read shows it. Label them `assumed` everywhere.
3. No caste or religion attributes anywhere.
4. Do not write "novel metric", "guarantee", "first", "state of the art".
5. Python only from `~/.venv/bin/python`. **Do not install packages.** If something is missing, tell the team. Installed: numpy, scipy, pandas, pyarrow, scikit-learn, xgboost, statsmodels, shap, fairlearn, mapie, fastapi, uvicorn, streamlit, plotly, matplotlib, pydantic, joblib, pytest. Not installed: lightgbm, pymysql, sqlalchemy.
6. Git messages: plain English describing the change. Never mention AI tools.
7. Never commit: `data/berka/`, `data/synth*/`, `data/fixtures/`, `artefacts/`, `*.sql`, `*.csv`.
8. All randomness: `numpy.random.default_rng(seed)`. Seeds are in `src/secondlook/config.py`.
9. The files and function signatures in `docs/SPLIT_CONTRACT.md` are frozen. If you must change one, message the other two first. Do not import another part's internal code, only the functions named in the contract.

## 3. Setup (each person, 10 minutes)

```bash
git clone https://github.com/Shreya-chaudhary1711/SecondLook.git
cd SecondLook
git fetch origin
git checkout -b <your-branch> origin/worktree-split-contract     # contract, shared schemas, fixtures
~/.venv/bin/python scripts/make_fixtures.py                      # random data in the real schema
export SL_DATA=data/fixtures/synth SL_ART=data/fixtures/artefacts
~/.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from secondlook.model.scorer import load_model; print(load_model().tau)"
```
The last line must print a number. Add `export PYTHONPATH=src` to your shell so `import secondlook` works everywhere. Run tests with `~/.venv/bin/python -m pytest tests -q`.

Fixture data are random numbers in the correct format. Build against them until Part 1 publishes the real data. When it does: `unset SL_DATA SL_ART`.

## 4. Repository layout and who owns what

```
src/secondlook/
  config.py contracts.py        shared, frozen
  berka/ sim/ model/ api/       Part 1
  audit/ uncertainty/           Part 2
  recourse/                     Part 3
app/Home.py                     Part 1
app/pages/1_Applicant.py        Part 1
app/pages/2_Audit.py            Part 2
app/pages/3_Recourse.py         Part 3
app/pages/4_Uncertainty.py      Part 2
app/pages/5_Data_and_Limits.py  Part 1
scripts/                        Part 1 (make_data.py, train.py), Part 2 (run_audit.py), Part 3 (run_recourse.py)
tests/                          each part adds test_<part>_*.py
docs/                           contract, specs, guide; Part 3 adds MODEL_CARD.md and DATASHEET.md
```
Touch only your own directories. Every `__init__.py` is empty and you create it when you make a new package directory.

## 5. Order of work and handoffs

Clock times are for 31 Oct. Adjust if you start late; the fixed points are not negotiable.

| Time | Part 1 | Part 2 | Part 3 |
|---|---|---|---|
| now to +2 h | simulator, `make_data.py`, **publish real `data/synth` (handoff A)** | group table and CIs, burden, tests on fixtures | income process, actions, Monte Carlo on fixtures |
| +2 to +5 h | train, calibrate, cost tau, `cash_features`, reasons; **publish `model.joblib` (handoff B)** | level-down, proxy card, conformal | plan optimiser, stress tests |
| +5 to +8 h | API, Applicant and Data pages, Berka sanity run | no-label bounds, pool vs fresh, 2 pages | Recourse page, `run_recourse_by_group` |
| 8:30 AM | **FEATURE FREEZE** | | |
| 8:30 to 9:45 | merge, fix bugs, run merge checklist | | write README, model card, datasheet |
| 9:45 to 11:15 | demo recording support | | **README and 3-minute video** |
| 11:45 to 12:30 | **upload** (portal opens 12:00 PM) | | |
| 1:30 PM | hackathon ends | | |

Handoff A: real parquet tables in `data/synth/`. Handoff B: `artefacts/model.joblib`. After each handoff, post in the team chat: "handoff A done, branch X, commit Y". Others pull, `unset SL_DATA SL_ART`, rerun.

If you are behind, cut in this order: pool-vs-fresh, sensitivity slider, retraining stress test, no-label mode, proxy card. Never cut: group table, burden audit, conformal (basic), recourse (basic), README, video.

## 6. Daily routine for every part

1. Write the function. 2. Write a pytest next to it using fixtures. 3. Run the test. 4. Commit and push your branch every hour: `git add <your files> && git commit -m "..." && git push -u origin <your-branch>`.

## 7. Definition of done for each part

A part is done when: all its tests pass; its functions return exactly the shapes in `docs/SPLIT_CONTRACT.md`; its pages open without errors against both fixtures and real data; and every table shows `n`.

## 8. Merge steps (Part 1 leads, 8:30 AM)

```bash
git checkout -b main-merge origin/worktree-split-contract
git merge origin/part1-data-model origin/part2-audit origin/part3-recourse   # resolve conflicts, they should only be in tests/ or app/
unset SL_DATA SL_ART
~/.venv/bin/python scripts/make_data.py && ~/.venv/bin/python scripts/train.py
~/.venv/bin/python scripts/run_recourse.py && ~/.venv/bin/python scripts/run_audit.py
~/.venv/bin/python -m pytest tests -q
~/.venv/bin/streamlit run app/Home.py
```
Open every page. Start the API (`~/.venv/bin/uvicorn secondlook.api.main:app --port 8000`) and call each endpoint in the table in `docs/IMPLEMENTATION_SPEC.md` section 8 once. Then push to the `main` of the submission repo with the team leader's approval.

## 9. When something is unclear

The long reasoning is in `docs/IMPLEMENTATION_SPEC.md` and `docs/PROJECT_PLAN.md`. If this guide and the spec disagree, this guide wins for code, the contract wins for file and function shapes. Ask in the team chat before inventing a new column or key.
