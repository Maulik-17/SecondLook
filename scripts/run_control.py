"""Control run: the whole pipeline with the injected biases switched off. Writes to data/control/ only.
Usage: python scripts/run_control.py"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "SL_BIAS": "0",
       "SL_DATA": str(ROOT / "data/control/synth"), "SL_ART": str(ROOT / "data/control/artefacts")}
for script in ("make_data.py", "train.py", "run_audit.py"):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, env=env, check=True,
                   stdout=None if script != "train.py" else subprocess.DEVNULL)
print("control run written to data/control/artefacts")
