from pathlib import Path

import pytest

from secondlook import config

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (config.ART_DIR / "model.joblib").exists() or not (config.DATA_DIR / "features.parquet").exists(),
                                reason="needs model and data")


@pytest.mark.parametrize("page", ["app/Home.py", "app/pages/1_Applicant.py", "app/pages/5_Data_and_Limits.py"])
def test_page_runs(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
