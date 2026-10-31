"""Stress tests for a chosen plan, and the months-to-approval table by group."""
import json

import numpy as np

from .. import config as C
from .. import contracts as CT
from .context import get_ctx, get_person, get_retrained

K_RETRAIN = 500


def _p(person, plan_idx, horizon, K, tau, **kw):
    from .plan import sim
    return float(sim(person, plan_idx, horizon, K, tau, **kw)["success"].mean())


def stress_tests(person, chosen, horizon, tau, K=1000, rho_override_base=None):
    """Success probability of the chosen plan under: drift, threshold_shift, gaming, persistence, retrain.

    retrain: mean over the precomputed retrained scorers of P(approved within the horizon), each judged at its
    own tau (see retrain.py). Note: the drift test uses innovation sd = 1.2 * sigma and the base simulation uses
    innovation sd = sigma, which differs from the guide; this earlier approach is kept unchanged.
    """
    rb = dict(rho_override=rho_override_base)
    return {
        "drift": _p(person, chosen, horizon, K, tau, vol_mult=1.2, **rb),
        "threshold_shift": _p(person, chosen, horizon, K, max(tau - 0.02, 0.0), **rb),
        "gaming": _p(person, chosen, horizon, K, tau, on_time_effective=False, **rb),
        "persistence": min(_p(person, chosen, horizon, K, tau, rho_override=r) for r in (0.0, 0.45, 0.9)),
        "retrain": float(np.mean([_p(person, chosen, horizon, K_RETRAIN, float(s.tau), scorer=s, **rb)
                                  for s in get_retrained()])),
    }


def run_recourse_by_group(n_per_group=100, horizon=12, K=300, seed=C.SEED):
    """Moderate fixed plan (1,1,1,1) for rejected test people; median months-to-approval per group."""
    from .plan import sim
    c = get_ctx()
    rng = np.random.default_rng(seed)
    pop = c["people"].join(c["scores"][["decision"]])
    pop = pop[(pop.split == "test") & (pop.decision == 0)]
    tau = float(c["scorer"].tau)
    cache, rows = {}, []
    for attr, levels in CT.GROUPS.items():
        for lev in levels:
            sub = pop[pop[attr] == lev]
            ids = sub.index.values
            if len(ids) > n_per_group:
                ids = rng.choice(ids, n_per_group, replace=False)
            p50s = []
            for pid in ids:
                if pid not in cache:
                    first = sim(get_person(pid), (1, 1, 1, 1), horizon, K, tau)["first"]
                    v = float(np.quantile(first, .5, method="inverted_cdf"))
                    cache[pid] = v if np.isfinite(v) else None
                p50s.append(cache[pid])
            fin = [v for v in p50s if v is not None]
            rows.append({"attribute": attr, "level": lev if isinstance(lev, str) else int(lev), "n": len(p50s),
                         "median_months": float(np.median(fin)) if fin else None,
                         "never_share": float(1 - len(fin) / len(p50s)) if p50s else 0.0})
    out = {"groups": rows}
    C.ART_DIR.mkdir(parents=True, exist_ok=True)
    (C.ART_DIR / "recourse_by_group.json").write_text(json.dumps(out, indent=1))
    return out
