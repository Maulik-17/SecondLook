import numpy as np
from secondlook.model.features import cash_features


def test_constant():
    K, w = 2, 6
    f = cash_features(np.full((K, w), 1000.), np.full((K, w), 800.), np.full((K, w), 2000.), np.full(K, 200.))
    assert np.allclose(f["inflow_cv"], 0) and np.allclose(f["expense_to_income"], 0.8)
    assert np.allclose(f["emi_to_income"], 0.2) and np.allclose(f["dip_months"], 0)
    assert len(f) == 12
