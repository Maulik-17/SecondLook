import json

import pandas as pd
from fastapi import FastAPI, HTTPException

from .. import config
from ..contracts import FEATURES
from ..model import explain
from ..model.scorer import load_model
from .schemas import NoLabelRequest, RecourseRequest, ScoreRequest

DISCLAIMER = "Synthetic data; an audit-tool demonstration, not a lending decision tool."
app = FastAPI(title="SecondLook")
STATE: dict = {}


def _state():
    """Loaded on first use so the module imports without artefacts present."""
    if not STATE:
        STATE["scorer"] = load_model()
        STATE["features"] = pd.read_parquet(config.DATA_DIR / "features.parquet").set_index("id", drop=False)
        STATE["people"] = pd.read_parquet(config.DATA_DIR / "people.parquet").set_index("id", drop=False)
        STATE["hash"] = getattr(STATE["scorer"], "config_hash", "unknown")
    return STATE


@app.on_event("startup")
def _load():
    try:
        _state()
    except FileNotFoundError:
        pass   # /health still answers; other routes report the missing file


def _wrap(d: dict) -> dict:
    return {**d, "config_hash": _state()["hash"], "disclaimer": DISCLAIMER}


def _json(name: str):
    path = config.ART_DIR / name
    if not path.exists():
        raise HTTPException(501, "not ready")
    data = json.loads(path.read_text())
    return _wrap(data) if isinstance(data, dict) else {"data": data, "disclaimer": DISCLAIMER}


@app.get("/health")
def health():
    try:
        h = _state()["hash"]
    except FileNotFoundError:
        h = "no-model"
    return {"status": "ok", "config_hash": h, "disclaimer": DISCLAIMER}


def _confidence(p, groups_row):
    try:
        from ..uncertainty.conformal import load_conformal
        res = load_conformal().predict_set(p, groups_row)[0]
        return {"available": True, "set": [int(v) for v in res["set"]], "abstain": bool(res["abstain"])}
    except Exception:   # not delivered yet, or the artefact is missing: report no confidence information
        return {"available": False}


@app.post("/score")
def score(req: ScoreRequest):
    st = _state()
    s = st["scorer"]
    if req.person_id is not None:
        if req.person_id not in st["features"].index:
            raise HTTPException(404, "unknown person_id")
        x = st["features"].loc[[req.person_id]]
        grp = st["people"].loc[[req.person_id]]
        thin = int(grp.thin_file.iloc[0])
    elif req.features is not None:
        missing = [f for f in FEATURES if f not in req.features]
        if missing:
            raise HTTPException(422, f"missing features: {missing}")
        x = pd.DataFrame([{f: req.features[f] for f in FEATURES}])
        thin = int(req.features["months_observed"] <= 5)
        grp = pd.DataFrame([{"gender": "M", "urban_rural": "urban", "region": "R1", "occupation": "gig_driver",
                             "thin_file": thin, "shared_phone": 0, "document_poor": 0}])
    else:
        raise HTTPException(422, "give person_id or features")
    p = s.predict_proba(x)
    conf = _confidence(p, grp)
    if conf["available"] and conf["abstain"]:
        decision = "refer"
    else:
        decision = "approve" if p[0] <= s.tau else "decline"
    return _wrap({"p_default": float(p[0]), "decision": decision, "tau": float(s.tau),
                  "reasons": explain.reasons(s, x), "confidence": conf, "thin_file": bool(thin)})


@app.post("/recourse")
def recourse(req: RecourseRequest):
    try:
        from ..recourse.plan import plan
    except ImportError:
        raise HTTPException(501, "not ready")
    return _wrap(plan(req.person_id, req.horizon, req.target_prob, req.rho_override))


@app.get("/audit/summary")
def audit_summary():
    return _json("audit_summary.json")


@app.post("/audit/nolabel")
def audit_nolabel(req: NoLabelRequest):
    try:
        from ..audit.nolabel import nolabel_bounds
    except ImportError:
        raise HTTPException(501, "not ready")
    return _wrap(nolabel_bounds(pd.DataFrame(req.proxy_table), pd.DataFrame(req.approval_by_cell)))


@app.get("/uncertainty/conformal")
def uncertainty_conformal():
    return _json("uncertainty.json")


@app.get("/uncertainty/poolfresh")
def uncertainty_poolfresh():
    return _json("poolfresh.json")


@app.get("/data/dictionary")
def data_dictionary():
    path = config.ART_DIR / "data_dictionary.json"
    if not path.exists():
        raise HTTPException(501, "not ready")
    return _wrap({"features": json.loads(path.read_text())})
