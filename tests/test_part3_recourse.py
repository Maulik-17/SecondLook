"""Part 3 tests. Run against fixtures (SL_DATA / SL_ART) or the real data; fixtures are built if missing."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secondlook import config as C  # noqa: E402

if not (C.ART_DIR / "model.joblib").exists():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fixtures.py")], check=True)
    pytest.skip("fixtures were just built; set SL_DATA/SL_ART to data/fixtures/* and rerun", allow_module_level=True)

from secondlook.recourse import income_process as IP  # noqa: E402
from secondlook.recourse import actions as A  # noqa: E402
from secondlook.recourse.context import get_ctx, get_person  # noqa: E402
from secondlook.recourse.plan import plan, sim  # noqa: E402
from secondlook.recourse.stats import wilson  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return get_ctx()


@pytest.fixture(scope="module")
def rejected(ctx):
    s = ctx["scores"]
    ids = s[s.decision == 0].index[:6]
    assert len(ids) > 0
    return [int(i) for i in ids]


def test_ar1_recovery_and_shrinkage():
    rng = np.random.default_rng(3)
    x = np.zeros(200)
    for t in range(1, 200):
        x[t] = 0.5 * x[t - 1] + 0.2 * rng.standard_normal()
    f = IP.fit_ar1(x)
    assert abs(f["rho"] - 0.5) < 0.1 and abs(f["sigma"] - 0.2) < 0.03
    short = IP.fit_ar1(x[:3])
    pri = {"rho_mean": 0.4, "ls_mean": np.log(0.2)}
    _, rho_post, _ = IP.shrink(short, pri)
    assert abs(rho_post - 0.4) < 0.15  # strong shrinkage toward the prior at w = 3


def test_action_costs():
    assert A.plan_cost((0, 0, 0, 0), 12) == 0
    assert len(A.all_plans()) == 81
    assert A.plan_cost((0, 0, 0, 2), 12) == A.plan_cost((0, 0, 0, 2), 6)  # one-off


def test_wilson_order():
    lo, est, hi = wilson(8, 10)
    assert lo <= est <= hi


def test_common_random_numbers(rejected):
    p = get_person(rejected[0])
    a = sim(p, (1, 1, 1, 1), 6, 200, 0.2)
    b = sim(p, (1, 1, 1, 1), 6, 200, 0.2)
    assert np.array_equal(a["p_matrix"], b["p_matrix"])
    r1, r2 = plan(rejected[0], horizon=6), plan(rejected[0], horizon=6)
    assert r1 == r2


def test_deterministic_path(rejected):
    p = get_person(rejected[0])
    tau = get_ctx()["scorer"].tau
    r = sim(p, (0, 0, 0, 0), 6, 50, tau, rho_override=0.0, vol_mult=0.0)
    # fit_ar1 draws are scaled by vol_mult = 0, so every path is identical
    assert np.allclose(r["p_matrix"], r["p_matrix"][0])
    assert r["success"].all() or not r["success"].any()


def test_horizon_monotone(rejected):
    for pid in rejected[:3]:
        p = get_person(pid)
        tau = get_ctx()["scorer"].tau
        s6 = sim(p, (1, 1, 1, 1), 6, 300, tau)["success"].mean()
        s12 = sim(p, (1, 1, 1, 1), 12, 300, tau)["success"].mean()
        assert s12 >= s6 - 1e-9


def test_intensity_monotone(rejected):
    for pid in rejected[:3]:
        p = get_person(pid)
        tau = get_ctx()["scorer"].tau
        lo = sim(p, (0, 0, 0, 0), 12, 500, tau)["success"].mean()
        hi = sim(p, (2, 2, 2, 2), 12, 500, tau)["success"].mean()
        assert hi >= lo - 0.02


def test_infeasible_returns_best(rejected):
    infeasible = []
    for pid in rejected:
        r = plan(pid, horizon=3, target_prob=0.999)
        if r["infeasible"]:
            infeasible.append(r)
            assert r["plan"]  # the strongest plan, not an empty one
    # at least one of the first rejected people cannot reach 0.999 within 3 months in the fixtures or real data
    assert infeasible


def test_already_approved(ctx):
    s = ctx["scores"]
    pid = int(s[s.decision == 1].index[0])
    r = plan(pid)
    assert r["plan"] == [] and r["cost"] == 0 and r["p_success"] == [1.0, 1.0, 1.0]


def test_unknown_id():
    with pytest.raises(KeyError):
        plan(10 ** 9)


def test_contract_shape(rejected):
    r = plan(rejected[0], horizon=8)
    for k in ["person_id", "plan", "cost", "p_success", "months_to_approval_quantiles", "fan_chart", "stress", "fragile"]:
        assert k in r
    lo, est, hi = r["p_success"]
    assert lo <= est <= hi
    assert set(r["stress"]) == {"drift", "threshold_shift", "gaming", "persistence"}
    assert set(r["months_to_approval_quantiles"]) == {"p10", "p50", "p90"}
    for k in ("month", "q10", "q50", "q90"):
        assert len(r["fan_chart"][k]) == 8


def test_rho_slider_matters(rejected):
    diffs = []
    for pid in rejected:
        a = plan(pid, rho_override=0.0)["p_success"][1]
        b = plan(pid, rho_override=0.9)["p_success"][1]
        diffs.append(abs(a - b))
    assert max(diffs) > 0
