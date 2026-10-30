import pytest
from streamlit.testing.v1 import AppTest

from secondlook import config as C

ROOT = C.ROOT


@pytest.mark.parametrize("page", ["2_Audit.py", "4_Uncertainty.py"])
def test_page_runs(page):
    need = "audit_summary.json" if page.startswith("2") else "uncertainty.json"
    if not (C.ART_DIR / need).exists():
        pytest.skip("run scripts/run_audit.py first")
    at = AppTest.from_file(str(ROOT / "app" / "pages" / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("config_hash" in c.value for c in at.caption)
    assert any("demonstration" in c.value.lower() or "demonstration" in w.value.lower() for c in at.caption for w in at.warning)
