#!/bin/bash
# Full pipeline: data, model, recourse, audit. Set PYTHON to pick the interpreter (default: python).
# The audit takes about 5 minutes; set AUDIT_ARGS=--fast to skip the bootstrap.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH=src
"$PY" scripts/make_data.py
"$PY" scripts/train.py
"$PY" scripts/run_recourse.py
"$PY" scripts/run_audit.py ${AUDIT_ARGS:-}
echo "done: data in data/synth, model and results in artefacts/"
echo "start the app: PYTHONPATH=src streamlit run app/Home.py   (http://localhost:8501)"
echo "start the API: PYTHONPATH=src uvicorn secondlook.api.main:app --port 8000   (docs at http://localhost:8000/docs)"
