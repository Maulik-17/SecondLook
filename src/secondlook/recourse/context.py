"""Cached tables, scorer and income prior (module-level, keyed by data/artefact directories)."""
import numpy as np
import pandas as pd

from .. import config as C
from ..model.scorer import load_model
from . import income_process as IP
from . import montecarlo as MC

_CACHE = {}


def get_ctx():
    key = (str(C.DATA_DIR), str(C.ART_DIR))
    if _CACHE.get("key") != key:
        d = C.DATA_DIR
        people = pd.read_parquet(d / "people.parquet")
        feats = pd.read_parquet(d / "features.parquet").set_index("id")
        scores = pd.read_parquet(d / "scores.parquet").set_index("id")
        monthly = pd.read_parquet(d / "monthly.parquet")
        monthly = monthly[monthly.month <= C.HISTORY_MONTHS]
        _CACHE.clear()
        _CACHE.update(key=key, people=people.set_index("id", drop=False), feats=feats, scores=scores,
                      monthly=monthly.set_index("id").sort_index(), scorer=load_model(),
                      prior=IP.load_prior(monthly.reset_index() if monthly.index.name == "id" else monthly, people),
                      persons={})
    return _CACHE


def get_person(pid):
    c = get_ctx()
    pid = int(pid)
    if pid not in c["people"].index:
        raise KeyError(pid)
    if pid not in c["persons"]:
        person = MC.build_person(c["people"].loc[pid], c["feats"].loc[pid], c["monthly"].loc[[pid]].reset_index())
        person["fit"] = IP.fit_ar1(np.log(np.maximum(person["inc"], 1.0)))
        person["prior"] = c["prior"][person["occupation"]]
        person["p_now"] = float(c["scores"].loc[pid, "p_default"])
        c["persons"][pid] = person
    return c["persons"][pid]
