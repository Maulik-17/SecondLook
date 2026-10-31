"""Starts a real uvicorn server on a spare port and calls every API endpoint over HTTP.
Needs data/ and artefacts/ (see README). Usage: PYTHONPATH=src python scripts/check_api_live.py"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.load(r)


def main():
    root = Path(__file__).resolve().parents[1]
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "secondlook.api.main:app", "--port", str(PORT), "--log-level", "warning"], cwd=root)
    try:
        for _ in range(60):
            try:
                call("GET", "/health")
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server did not start")
        prox = [{"cell": 0, "p_a1": 0.2, "share": 0.5}, {"cell": 1, "p_a1": 0.7, "share": 0.5}]
        appr = [{"cell": 0, "approval_rate": 0.5}, {"cell": 1, "approval_rate": 0.8}]
        checks = [("GET", "/health", None), ("GET", "/data/dictionary", None), ("GET", "/audit/summary", None),
                  ("GET", "/uncertainty/conformal", None), ("GET", "/uncertainty/poolfresh", None),
                  ("POST", "/score", {"person_id": 38}), ("POST", "/recourse", {"person_id": 38, "horizon": 12}),
                  ("POST", "/audit/nolabel", {"proxy_table": prox, "approval_by_cell": appr})]
        bad = 0
        for m, p, b in checks:
            try:
                code, js = call(m, p, b)
            except Exception as ex:
                code, js = "ERR", str(ex)
            ok = code == 200
            bad += not ok
            print(f"{'ok ' if ok else 'FAIL'} {m:4s} {p:26s} {code}")
        print("ALL OK" if not bad else f"{bad} FAILED")
        sys.exit(1 if bad else 0)
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == "__main__":
    main()
