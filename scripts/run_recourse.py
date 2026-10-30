"""Writes ART_DIR/recourse_by_group.json. Usage: python scripts/run_recourse.py [n_per_group]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secondlook.recourse.stress import run_recourse_by_group

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = run_recourse_by_group(n_per_group=n)
    for g in out["groups"]:
        print(g)
