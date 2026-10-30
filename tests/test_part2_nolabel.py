import numpy as np
import pandas as pd

from secondlook.audit.nolabel import nolabel_bounds


def _tables(p_a1, ap):
    k = len(p_a1)
    return (pd.DataFrame({"cell": range(k), "p_a1": p_a1, "share": np.ones(k) / k}),
            pd.DataFrame({"cell": range(k), "approval_rate": ap}))


def test_pure_cells_collapse():
    b = nolabel_bounds(*_tables([0.0, 1.0], [0.2, 0.8]))
    assert b["width"] < 1e-6 and abs(b["lower"] - 0.6) < 1e-6


def test_uniform_cells_wide():
    b = nolabel_bounds(*_tables([0.5, 0.5], [0.2, 0.8]))
    assert b["width"] > 0.3


def test_truth_inside_interval():
    r = np.random.default_rng(0)
    for _ in range(100):
        k = 4
        pa1 = r.uniform(0.1, 0.9, k)
        share = r.dirichlet(np.ones(k) * 5)
        n = 20000
        z = r.choice(k, n, p=share)
        a = r.random(n) < pa1[z]
        y = r.random(n) < np.where(a, 0.7, 0.5) * (0.8 + 0.1 * z)
        truth = y[a].mean() - y[~a].mean()
        prox = pd.DataFrame({"cell": range(k), "p_a1": [a[z == i].mean() for i in range(k)],
                             "share": [(z == i).mean() for i in range(k)]})
        appr = pd.DataFrame({"cell": range(k), "approval_rate": [y[z == i].mean() for i in range(k)]})
        b = nolabel_bounds(prox, appr)
        assert b["lower"] - 1e-9 <= truth <= b["upper"] + 1e-9
