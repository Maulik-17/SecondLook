import pytest
from fastapi.testclient import TestClient

from secondlook import config
from secondlook.contracts import FEATURES

pytestmark = pytest.mark.skipif(not (config.ART_DIR / "model.joblib").exists() or not (config.DATA_DIR / "features.parquet").exists(),
                                reason="needs model and data (set SL_DATA / SL_ART)")


@pytest.fixture(scope="module")
def client():
    from secondlook.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_person(client):
    r = client.post("/score", json={"person_id": 0})
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["p_default"] <= 1 and len(j["reasons"]) == 5
    assert j["decision"] in ("approve", "refer", "decline") and "disclaimer" in j and "config_hash" in j
    assert not any(x["feature"] in ("night_active_share", "local_market_share") for x in j["reasons"])


def test_score_features_and_422(client):
    from secondlook.api.main import _state
    row = _state()["features"].iloc[0]
    feats = {f: float(row[f]) for f in FEATURES}
    assert client.post("/score", json={"features": feats}).status_code == 200
    feats.pop("inflow_cv")
    assert client.post("/score", json={"features": feats}).status_code == 422
    assert client.post("/score", json={}).status_code == 422


def test_dictionary_and_unready(client):
    d = client.get("/data/dictionary").json()
    assert len(d["features"]) == len(FEATURES)
    code = client.post("/recourse", json={"person_id": 0}).status_code
    assert code in (200, 501)
    assert client.get("/audit/summary").status_code in (200, 501)
