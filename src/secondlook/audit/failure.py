"""Failure-mode panel: level-down, inversion, gaming, infeasible share, referral view."""
import numpy as np

from .. import config as C
from ..contracts import GROUPS
from . import burden as B

EXTRACTION_NOTE = "The recourse endpoint reveals score sensitivities; this is a model-extraction risk."


def level_down_curve(df, ctx, tau_grid=None):
    grid = np.linspace(0.03, 0.5, 20) if tau_grid is None else tau_grid
    out = []
    for t in grid:
        o = B.person_efforts(df, ctx, float(t), "base")
        rg = B.rg_mask(df, o["rejected"])
        rows = B.burden_rows(df, o["effort"], rg, bootstrap=False)
        h = B.headline(rows)
        ok = [r["burden"][1] for r in rows if r["n"] >= C.MIN_GROUP_N and np.isfinite(r["burden"][1])]
        out.append({"tau": float(t), "approval_rate": float((~o["rejected"]).mean()),
                    "worst_group_burden": float(max(ok)) if ok else float("nan"), "ratio": h["ratio"]})
    return out


def level_down_flag(curve, current):
    """True when a lower-approval tau (more than 5 points lower) cuts the worst/best burden ratio by >10%."""
    r0 = current["ratio"]
    if r0 is None or not np.isfinite(r0):
        return False
    for c in curve:
        r = c["ratio"]
        if r is not None and np.isfinite(r) and c["approval_rate"] < current["approval_rate"] - 0.05 and r < 0.9 * r0:
            return True
    return False


def failure_panel(df, ctx, tau, current_row, o, curve, conformal_abstain=None):
    rg = B.rg_mask(df, o["rejected"])
    gi = B.gaming_and_inversion(df, ctx, o)
    fin = o["effort"][o["rejected"] & np.isfinite(o["effort"])]
    inf_share = float((o["rejected"] & ~np.isfinite(o["effort"])).sum() / max(o["rejected"].sum(), 1))
    panel = {
        "level_down": level_down_flag(curve, current_row),
        "inversion": bool(gi["inversion_count"] > 0), "inversion_count": gi["inversion_count"],
        "gaming": bool(gi["gaming_ratio"] > 0.3), "gaming_ratio": gi["gaming_ratio"],
        "gaming_note": "Rent, utility and recharge signals are assumed, not shown to cause default.",
        "infeasible_share": inf_share,
        "effort_p95": float(np.percentile(fin, 95)) if len(fin) else None,
        "extraction_note": EXTRACTION_NOTE,
    }
    if conformal_abstain is not None:
        # share of the creditworthy per group who are rejected OR referred to a person
        view = []
        good = (df.y_true == 0).values
        adverse = o["rejected"] | conformal_abstain
        for a, levels in GROUPS.items():
            for l in levels:
                m = (df[a] == l).values & good
                view.append({"attribute": a, "level": B._py(l), "n_good": int(m.sum()),
                             "rejected_share": float(o["rejected"][m].mean()) if m.any() else None,
                             "rejected_or_referred_share": float(adverse[m].mean()) if m.any() else None})
        panel["referral_view"] = view
    return panel
