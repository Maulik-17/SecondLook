#!/bin/bash
# Control run: the whole pipeline with the injected biases switched off. Writes to data/control/ only.
# Set PYTHON to pick the interpreter (default: python).
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH=src SL_BIAS=0 SL_DATA="$PWD/data/control/synth" SL_ART="$PWD/data/control/artefacts"
"$PY" scripts/make_data.py
"$PY" scripts/train.py > /dev/null
"$PY" scripts/run_audit.py
echo "control run written to data/control/artefacts"
