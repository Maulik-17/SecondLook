"""One command for the whole project. Builds data, model and audit results if they are missing, then serves the
Streamlit app (http://localhost:8501) and the API (http://localhost:8000/docs). Ctrl-C stops both.
Usage: python scripts/start.py [--fast]     (--fast skips the audit bootstrap when building)"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import ROOT, run_all  # noqa: E402


def main() -> None:
    if not (ROOT / "artefacts" / "audit_summary.json").exists() or not (ROOT / "data" / "synth" / "people.parquet").exists():
        print("building data, model and audit results (about 7 minutes; --fast shortens it)")
        run_all("--fast" in sys.argv[1:])
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "secondlook.api.main:app", "--port", "8000", "--log-level", "warning"],
                           cwd=ROOT, env=env)
    print("API:  http://localhost:8000/docs\nApp:  http://localhost:8501")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/Home.py", "--server.port", "8501"], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        pass
    finally:
        api.terminate()


if __name__ == "__main__":
    main()
