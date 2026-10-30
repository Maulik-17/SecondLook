import numpy as np
import pytest
from scipy.optimize import minimize

from secondlook.audit.burden import effort_core, compute_efforts, threshold_for
from secondlook.model.scorer import load_model


def test_closed_form_matches_line_search():
    a, sd, W = np.array([0.8, -0.5, 1.2]), np.array([1.5, 2.0, 0.7]), np.array([2.0, 1.0, 3.0])
    c = a * sd
    slack = 1.3
    e, dz = effort_core(c, W, np.array([slack]))
    res = minimize(lambda z: np.sqrt(np.sum(W * z ** 2)), np.zeros(3), method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda z: -(c @ z) - slack}], options={"ftol": 1e-12})
    assert abs(res.fun - e[0]) < 1e-3
    assert np.allclose(res.x, dz[0], atol=1e-3)


def test_zero_slack_zero_effort():
    out = compute_efforts(np.zeros((1, 2)), np.array([0.0]), np.array([1.0, 1.0]), np.ones(2),
                          -np.ones(2) * 9, np.ones(2) * 9, 0.0, np.ones(2))
    assert out["effort"][0] == 0


def test_equal_weights_effort_is_slack_over_norm():
    a, sd = np.array([1.0, 2.0, -2.0]), np.array([1.0, 0.5, 2.0])
    e, _ = effort_core(a * sd, np.ones(3), np.array([2.0]))
    assert abs(e[0] - 2.0 / np.linalg.norm(a * sd)) < 1e-12


def test_infeasible_is_inf():
    out = compute_efforts(np.zeros((1, 2)), np.array([100.0]), np.array([1.0, 1.0]), np.ones(2),
                          -np.ones(2), np.ones(2), 0.0, np.ones(2))
    assert np.isinf(out["effort"][0])


def test_threshold_for_matches_scorer():
    try:
        s = load_model()
    except Exception:
        pytest.skip("no model")
    assert abs(threshold_for(s, s.tau) - s.logit_threshold()) < 1e-9
