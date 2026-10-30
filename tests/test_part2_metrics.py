import numpy as np
import pandas as pd

from secondlook.audit.metrics import wilson, boot, boot_mean, ece, group_table


def test_wilson():
    lo, est, hi = wilson(50, 100)
    assert lo < 0.5 < hi and abs(lo - 0.404) < 0.002 and est == 0.5
    assert np.isnan(wilson(0, 0)[1])


def test_boot_coverage():
    covered = 0
    for s in range(100):
        x = np.random.default_rng(1000 + s).normal(5, 2, 60)
        if s < 20:
            lo, est, hi = boot(lambda d: d.x.mean(), pd.DataFrame({"x": x}), B=200, seed=s)
        else:
            lo, est, hi = boot_mean(x, B=200, seed=s)
        covered += lo <= 5 <= hi
    assert covered >= 90


def test_ece_calibrated():
    p = np.repeat([0.1, 0.9], 1000)
    y = (np.random.default_rng(0).random(2000) < p).astype(int)
    assert ece(p, y) < 0.05


def test_group_table_shape():
    from secondlook.contracts import GROUPS
    r = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({a: r.choice(l, n) for a, l in GROUPS.items()})
    df["p_default"] = r.random(n) * 0.5
    df["y_true"] = (r.random(n) < df.p_default).astype(int)
    df["observed"] = (r.random(n) < 0.5).astype(int)
    rows = group_table(df, 0.2)
    assert len(rows) == sum(len(v) for v in GROUPS.values())
    for k in ("n", "approval", "base_rate", "tpr", "fpr", "ppv", "ece", "brier", "missing_outcome_share"):
        assert k in rows[0]
    assert sum(x["n"] for x in rows if x["attribute"] == "gender") == n
