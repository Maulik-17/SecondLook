import json

import numpy as np

from .. import config as C
from ..model.scorer import load_model
from . import burden as B
from .data import DISCLAIMER, load, test_pop
from .intersections import scan as intersection_scan
from .failure import level_down_curve, failure_panel
from .metrics import group_table, disparate_impact
from .nolabel import nolabel_summary
from .proxy import proxy_leakage
from .tradeoff import tradeoff


def _abstain(te):
    """Conformal abstentions on the evaluation rows, or None when no conformal model has been saved."""
    try:
        from ..uncertainty.conformal import load_conformal
        cm = load_conformal()
        res = cm.predict_set(te.p_default.values, te[cm.cells])
        return np.array([r["abstain"] for r in res])
    except Exception:
        return None


def is_quiet(out):
    """Null-world check: no failure flag, no burden gap left after matching on true risk, and the planted gender
    proxy is not the top predictor. (The raw worst/best burden ratio is reported but not used here: it follows
    base-rate differences, which the null world has on purpose.)"""
    gender = [p for p in out["proxy_leakage"] if p["attribute"] == "gender"]
    return bool(not any(out["failure_flags"].values()) and not out["failure_panel"]["unexplained_gap"]
                and not any(p["planted_top"] for p in gender))


def run_audit(tau, bootstrap=True, df=None, scorer=None, write=True, with_proxy=True, B_boot=None):
    df = load() if df is None else df
    scorer = load_model() if scorer is None else scorer
    te = test_pop(df)
    Bn = (B_boot or (C.BOOTSTRAP_B if bootstrap else 0)) if bootstrap else 0
    Bb = min(Bn, 200) if Bn else 0
    ctx = B.Context(df, scorer)

    groups = group_table(te, tau, bootstrap=bootstrap)
    o = B.person_efforts(te, ctx, tau, "base")
    rg = B.rg_mask(te, o["rejected"])
    brows = B.burden_rows(te, o["effort"], rg, B=max(Bb, 1), bootstrap=bootstrap)
    head = B.headline(brows)
    curve = level_down_curve(te, ctx)
    cur_h = {"ratio": head["ratio"], "approval_rate": float((~o["rejected"]).mean())}
    rm = B.risk_matched_gap(te, o["effort"], o["rejected"], brows, B=max(Bb, 50))
    panel = failure_panel(te, ctx, tau, cur_h, o, curve, _abstain(te), unexplained=B.unexplained_gap(rm))
    sens = B.sensitivity(te, ctx, tau)
    sets = {}
    for w in B.WEIGHTS:
        ow = B.person_efforts(te, ctx, tau, w)
        sets[w] = B.burden_rows(te, ow["effort"], B.rg_mask(te, ow["rejected"]), bootstrap=False)

    out = {
        "tau": float(tau), "config_hash": str(getattr(scorer, "config_hash", "unset")), "disclaimer": DISCLAIMER,
        "evaluation_population": "test split", "n_eval": int(len(te)),
        "groups": groups, "disparate_impact": disparate_impact(groups),
        "disparate_impact_note": "The four-fifths rule is US employment guidance and has no legal force in India.",
        "burden": brows, "burden_headline": head,
        "burden_note": "Burden among the creditworthy who were rejected; effort uses assumed weights and is a diagnostic. For a linear scorecard the weight sets rescale every person's effort by a common factor unless a feature range limit binds, so weight sensitivity is expected to be small.",
        "burden_by_weight_set": sets, "weight_sensitivity": sens,
        "budget_curve": B.budget_curves(te, o["effort"], rg),
        "risk_matched_gap": rm,
        "level_down": curve,
        "failure_flags": {k: panel[k] for k in ("level_down", "inversion", "gaming")},
        "failure_panel": panel,
        "null_world_quiet": None,
        "proxy_leakage": proxy_leakage(df, scorer, tau, B=Bb or 50, n_perm=100 if bootstrap else 20, fast=not bootstrap) if with_proxy else [],
        "tradeoff": tradeoff(groups),
        "nolabel": nolabel_summary(te, tau),
        "months_to_approval": None,
        "intersections": intersection_scan(te, o["effort"], rg, o["rejected"], B=min(Bb, 100)),
    }
    f = C.ART_DIR / "recourse_by_group.json"
    if f.exists():
        out["months_to_approval"] = json.load(open(f))["groups"]
    if write:
        C.ART_DIR.mkdir(parents=True, exist_ok=True)
        with open(C.ART_DIR / "audit_summary.json", "w") as fh:
            json.dump(out, fh, default=float)
    return out


def null_world_check(data_dir, scorer=None, tau=None):
    """Run the audit on another data folder (e.g. data/synth_null) with no bootstrap; returns is_quiet(...)."""
    scorer = scorer or load_model()
    out = run_audit(tau if tau is not None else scorer.tau, bootstrap=False, df=load(data_dir, scorer), scorer=scorer, write=False)
    return is_quiet(out), out
