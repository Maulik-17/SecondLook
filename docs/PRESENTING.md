# Showing SecondLook

Every team member (Harshit Jangir, Nikhil Saini, Shreya, Maulik Pratap Singhal) is a co-author and can present, host or fork this project under Apache-2.0. Keep `LICENSE` and `NOTICE` with any copy.

## Run it yourself in a few minutes

```
git clone https://github.com/Maulik-17/SecondLook.git && cd SecondLook
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt     # Python 3.14
bash scripts/start.sh        # builds data and model on first run (about 7 minutes; FAST=1 shortens it)
```

The app opens at http://localhost:8501 and the API at http://localhost:8000/docs. For a live demo, run it once beforehand so the data is built, and open each page once so nothing is cold.

To host your own copy, see `docs/DEPLOY.md`.

## A 3-minute walk-through

`docs/DEMO_SCRIPT.md` has the full script. In short: the Applicant page (a score with reasons and a referral when unsure), the Audit page (burden and the failure flags), the Recourse page (probability of approval over income futures), then the limits page.

## How to describe it accurately

- "We built SecondLook, an audit-first credit scoring tool for thin-file applicants, as a team of four at SPHINX'26 and won first place." Then say what you personally worked on.
- It runs on synthetic data calibrated on the Czech Berka dataset. It is an audit-tool demonstration, not a lending decision tool and not a finding about real lenders.
- The three-seed control run showed the audit's flags are a sensitive screen, not a detector of the injected biases. Say so if asked (`docs/CONTROL_RUN.md`).
- Do not describe it as solo work.
