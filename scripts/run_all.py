"""Full pipeline: data, model, recourse, audit. Usage: python scripts/run_all.py [--fast]
The audit takes about 5 minutes; --fast skips its bootstrap."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_all(fast: bool = False) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    steps = [["make_data.py"], ["train.py"], ["run_recourse.py"], ["run_audit.py"] + (["--fast"] if fast else [])]
    for step in steps:
        subprocess.run([sys.executable, str(ROOT / "scripts" / step[0]), *step[1:]], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    run_all("--fast" in sys.argv[1:])
    print("done: data in data/synth, model and results in artefacts/")
    print("start everything: python scripts/start.py")
