#!/bin/bash
# One command for the whole project: builds data, model and audit results if they are missing, then serves
# the Streamlit app (http://localhost:8501) and the API (http://localhost:8000/docs). Ctrl-C stops both.
# Usage: bash scripts/start.sh            (PYTHON picks the interpreter, default python; FAST=1 skips the audit bootstrap)
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH=src
if [ ! -f artefacts/audit_summary.json ] || [ ! -f data/synth/people.parquet ]; then
  echo "building data, model and audit results (about 7 minutes; FAST=1 shortens it)"
  PYTHON="$PY" AUDIT_ARGS="${FAST:+--fast}" bash scripts/run_all.sh
fi
"$PY" -m uvicorn secondlook.api.main:app --port 8000 --log-level warning &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT INT TERM
echo "API:  http://localhost:8000/docs"
echo "App:  http://localhost:8501"
"$PY" -m streamlit run app/Home.py --server.port 8501
