"""Control run over several seeds: for each seed, the pipeline with the injected biases on and off (audit in --fast mode).
Writes to data/seeds/<seed>_bias<0|1>/ and prints one summary line per run.
Usage: python scripts/run_seeds.py [seed ...]     (default seeds 101 202 303)"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_one(seed: str, bias: int) -> subprocess.Popen:
    d = ROOT / "data" / "seeds" / f"{seed}_bias{bias}"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "SL_SEED": seed, "SL_BIAS": str(bias),
           "SL_DATA": str(d / "synth"), "SL_ART": str(d / "artefacts")}
    code = "import subprocess,sys;" + "".join(
        f"subprocess.run([sys.executable,'scripts/{s[0]}',*{s[1:]!r}],check=True,stdout=subprocess.DEVNULL);"
        for s in (["make_data.py"], ["train.py"], ["run_audit.py", "--fast"]))
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env)


def main(seeds: list[str]) -> None:
    for s in seeds:
        procs = [run_one(s, b) for b in (1, 0)]
        for p in procs:
            if p.wait() != 0:
                sys.exit(f"run failed for seed {s}")
    for s in seeds:
        for b in (1, 0):
            d = ROOT / "data" / "seeds" / f"{s}_bias{b}" / "artefacts"
            a, m = json.load(open(d / "audit_summary.json")), json.load(open(d / "metrics.json"))
            print(f"seed {s} bias {'on ' if b else 'off'} tau {a['tau']:.2f} auc {m['test']['auc'][1]:.3f} "
                  f"burden_ratio {a['burden_headline']['ratio']:.2f} flags {sum(a['failure_flags'].values())} "
                  f"unexplained_gap {a['failure_panel']['unexplained_gap']} null_quiet {a['null_world_quiet']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["101", "202", "303"])
