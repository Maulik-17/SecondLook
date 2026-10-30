"""Candidate behaviour changes. Levels and costs are ASSUMPTIONS (effort points); replace via set_actions()."""
import itertools

ACTIONS = {
    "cut_spend":    {"levels": [0.00, 0.05, 0.10], "cost": [0.0, 1.0, 2.5]},   # share of variable spending cut
    "extra_income": {"levels": [0.00, 0.05, 0.10], "cost": [0.0, 2.0, 4.5]},   # extra inflow, share of income
    "on_time":      {"levels": [0.00, 0.10, 0.20], "cost": [0.0, 0.5, 1.0]},   # rise in rent/utility/recharge on-time rate (ASSUMED signal)
    "smaller_loan": {"levels": [0.00, 0.15, 0.30], "cost": [0.0, 0.8, 2.0]},   # one-off cut in requested loan
}
NAMES = list(ACTIONS)


def set_actions(new):
    """Replace the action table (same structure as ACTIONS)."""
    ACTIONS.clear(); ACTIONS.update(new)
    NAMES[:] = list(ACTIONS)


def _idx(levels_idx):
    if isinstance(levels_idx, (tuple, list)):
        return dict(zip(NAMES, levels_idx))
    return levels_idx


def plan_cost(levels_idx, horizon):
    """Monthly costs times horizon, except smaller_loan which is one-off."""
    tot = 0.0
    for a, i in _idx(levels_idx).items():
        c = ACTIONS[a]["cost"][i]
        tot += c if a == "smaller_loan" else c * horizon
    return float(tot)


def all_plans():
    return list(itertools.product(range(3), repeat=len(NAMES)))


def plan_levels(levels_idx):
    d = _idx(levels_idx)
    return {a: ACTIONS[a]["levels"][d[a]] for a in NAMES}


def describe(levels_idx, horizon):
    out = []
    for a, i in _idx(levels_idx).items():
        if i > 0:
            c = ACTIONS[a]["cost"][i]
            out.append({"action": a, "amount": ACTIONS[a]["levels"][i],
                        "cost": float(c if a == "smaller_loan" else c * horizon)})
    return out
