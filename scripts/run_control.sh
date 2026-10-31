#!/bin/bash
# Control run: the whole pipeline with the injected biases switched off. Writes to data/control/ only.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src SL_BIAS=0 SL_DATA="$PWD/data/control/synth" SL_ART="$PWD/data/control/artefacts"
python scripts/make_data.py
python scripts/train.py > /dev/null
python scripts/run_audit.py
echo "control run written to data/control/artefacts"
