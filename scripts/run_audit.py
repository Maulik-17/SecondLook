"""Rebuild every Part 2 JSON: uncertainty.json, poolfresh.json, audit_summary.json.
Usage: python scripts/run_audit.py [--fast]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secondlook import config as C
from secondlook.model.scorer import load_model
from secondlook.audit.report import run_audit, null_world_check
from secondlook.uncertainty.conformal import run_uncertainty
from secondlook.uncertainty.pool_vs_fresh import run_poolfresh

if __name__ == "__main__":
    fast = "--fast" in sys.argv
    s = load_model()
    run_uncertainty(C.EXPLORATION_EPS, C.ALPHA)          # first, so the audit can use the saved conformal model
    for e in (C.EXPLORATION_EPS,):
        run_poolfresh(e, rounds=20)
    out = run_audit(s.tau, bootstrap=not fast)
    null_dir = C.DATA_DIR.parent / "synth_null"
    if null_dir.exists():
        out["null_world_quiet"] = null_world_check(null_dir, s)[0]
        with open(C.ART_DIR / "audit_summary.json", "w") as fh:
            json.dump(out, fh, default=float)
    print("audit_summary.json written | tau", round(s.tau, 3), "| flags", out["failure_flags"],
          "| burden ratio", out["burden_headline"]["ratio"], "| null_world_quiet", out["null_world_quiet"])
