import numpy as np
import pandas as pd

from secondlook.berka.features import build


def test_temporal_synthetic():
    """Transactions on or after the loan date must never influence the features."""
    d = pd.date_range("1996-01-01", periods=300, freq="D")
    tr = pd.DataFrame({"trans_id": np.arange(300), "account_id": 1, "date": d.strftime("%Y-%m-%d"),
                       "type": np.where(np.arange(300) % 2 == 0, "PRIJEM", "VYDAJ"), "amount": 100.0, "balance": 5000.0})
    loan = pd.DataFrame({"loan_id": [1], "account_id": [1], "date": ["1996-09-01"], "payments": [500.0], "status": ["B"]})
    base = build(loan, tr)
    tr2 = tr.copy()
    tr2.loc[tr2.date >= "1996-09-01", ["amount", "balance"]] = 9e6        # poison the future
    after = build(loan, tr2)
    assert (base["date_used_max"] < pd.Timestamp("1996-09-01")).all()
    pd.testing.assert_frame_equal(base, after)
    assert base.y.iloc[0] == 1
