"""Entry point of the packaged (PyInstaller) build: starts the API and the Streamlit app, opens the browser.

The build folder holds this executable next to app/, src/, data/, artefacts/ and .streamlit/. The working
directory is set to that folder so the relative paths used by the app resolve.
"""
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_PORT = int(os.environ.get("SL_APP_PORT", "8501"))
API_PORT = int(os.environ.get("SL_API_PORT", "8000"))


def home() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]


def main() -> None:
    root = home()
    os.chdir(root)
    os.environ.setdefault("SL_DATA", str(root / "data" / "synth"))
    os.environ.setdefault("SL_ART", str(root / "artefacts"))
    sys.path.insert(0, str(root / "src"))
    if not (root / "artefacts" / "metrics.json").exists():
        sys.exit(f"Missing {root / 'artefacts' / 'metrics.json'}: the package is incomplete.")

    import uvicorn
    from secondlook.api.main import app as api_app

    server = uvicorn.Server(uvicorn.Config(api_app, host="127.0.0.1", port=API_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()

    def open_browser():
        time.sleep(4)
        webbrowser.open(f"http://localhost:{APP_PORT}")
    if os.environ.get("SL_NO_BROWSER") != "1":
        threading.Thread(target=open_browser, daemon=True).start()

    from streamlit.web import bootstrap
    flags = {"server.port": APP_PORT, "server.headless": True, "global.developmentMode": False,
             "browser.gatherUsageStats": False}
    print(f"SecondLook: app http://localhost:{APP_PORT}  |  API http://localhost:{API_PORT}/docs  |  close this window to stop")
    bootstrap.run(str(root / "app" / "Home.py"), False, [], flags)


if __name__ == "__main__":
    main()
