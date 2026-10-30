import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondlook import config as C  # noqa: E402

pytest.importorskip("streamlit")
if not (C.ART_DIR / "model.joblib").exists():
    pytest.skip("no model artefact", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_recourse_page_runs():
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "3_Recourse.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Total cost" in m.label for m in at.metric)
