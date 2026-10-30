"""Cheapest plan whose probability of reaching approval within the horizon is at least the target."""
import numpy as np

from . import actions as A
from . import montecarlo as MC
from .context import get_ctx, get_person
from .stats import wilson

SEED = 20261031
K_SEARCH, K_FINAL = 500, 1000
FRAGILE_BELOW = 0.6


def _q(first, q):
    v = float(np.quantile(first, q, method="inverted_cdf"))
    return None if not np.isfinite(v) else v


def sim(person, plan_idx, horizon, K, tau, **kw):
    c = get_ctx()
    return MC.simulate(person, person["fit"], person["prior"], plan_idx, horizon, K, tau, c["scorer"], SEED, **kw)


def plan(person_id, horizon=12, target_prob=0.8, rho_override=None, tau=None):
    c = get_ctx()
    person = get_person(person_id)  # KeyError for unknown id
    tau = float(tau if tau is not None else c["scorer"].tau)
    base = {"person_id": int(person_id), "plan": [], "cost": 0.0, "infeasible": False}
    if person["p_now"] <= tau:  # already approved
        h = list(range(1, horizon + 1))
        return {**base, "p_success": [1.0, 1.0, 1.0],
                "months_to_approval_quantiles": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
                "fan_chart": {"month": h, "q10": [person["p_now"]] * horizon,
                              "q50": [person["p_now"]] * horizon, "q90": [person["p_now"]] * horizon},
                "stress": {"drift": 1.0, "threshold_shift": 1.0, "gaming": 1.0, "persistence": 1.0},
                "fragile": False}
    plans = sorted(A.all_plans(), key=lambda p: (A.plan_cost(p, horizon), p))
    kw = dict(rho_override=rho_override)
    top = max(plans, key=lambda p: (sum(p), A.plan_cost(p, horizon)))
    p_top = sim(person, top, horizon, K_SEARCH, tau, **kw)["success"].mean()
    infeasible = p_top < target_prob
    if infeasible:
        chosen = top
    else:
        chosen = top
        for pl in plans:
            if sim(person, pl, horizon, K_SEARCH, tau, **kw)["success"].mean() >= target_prob:
                chosen = pl
                break
    r = sim(person, chosen, horizon, K_FINAL, tau, **kw)
    pm = r["p_matrix"]
    p_succ = wilson(int(r["success"].sum()), K_FINAL)
    from .stress import stress_tests
    st = stress_tests(person, chosen, horizon, tau, rho_override_base=rho_override)
    return {
        "person_id": int(person_id), "plan": A.describe(chosen, horizon), "cost": A.plan_cost(chosen, horizon),
        "p_success": p_succ,
        "months_to_approval_quantiles": {k: _q(r["first"], q) for k, q in (("p10", .1), ("p50", .5), ("p90", .9))},
        "fan_chart": {"month": list(range(1, horizon + 1)),
                      "q10": np.quantile(pm, .1, axis=0).tolist(), "q50": np.quantile(pm, .5, axis=0).tolist(),
                      "q90": np.quantile(pm, .9, axis=0).tolist()},
        "stress": st,
        "fragile": bool(p_succ[1] >= target_prob and min(st.values()) < FRAGILE_BELOW),
        "infeasible": bool(infeasible), "tau": tau,
    }
