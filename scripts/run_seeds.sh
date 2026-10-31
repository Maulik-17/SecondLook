#!/bin/bash
# Control run over several seeds: for each seed, the pipeline with the injected biases on and off (audit in --fast mode).
# Writes to data/seeds/<seed>_bias<0|1>/ and prints one summary line per run. Usage: scripts/run_seeds.sh [seed ...]
# Set PYTHON to pick the interpreter (default: python).
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
SEEDS="${@:-101 202 303}"
export PYTHONPATH=src
for s in $SEEDS; do
  for b in 1 0; do
    d="$PWD/data/seeds/${s}_bias${b}"
    (
      export SL_SEED=$s SL_BIAS=$b SL_DATA="$d/synth" SL_ART="$d/artefacts"
      "$PY" scripts/make_data.py > /dev/null && "$PY" scripts/train.py > /dev/null && "$PY" scripts/run_audit.py --fast > /dev/null
    ) &
  done
  wait
done
"$PY" - <<EOF
import json
for s in "$SEEDS".split():
    for b in (1, 0):
        d = f"data/seeds/{s}_bias{b}/artefacts"
        a, m = json.load(open(f"{d}/audit_summary.json")), json.load(open(f"{d}/metrics.json"))
        print(f"seed {s} bias {'on ' if b else 'off'} tau {a['tau']:.2f} auc {m['test']['auc'][1]:.3f} burden_ratio {a['burden_headline']['ratio']:.2f} "
              f"flags {sum(a['failure_flags'].values())} unexplained_gap {a['failure_panel']['unexplained_gap']} null_quiet {a['null_world_quiet']}")
EOF
