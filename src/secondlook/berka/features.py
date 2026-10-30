"""Berka loan-level cash features. Uses only transactions strictly before the loan date."""
import numpy as np
import pandas as pd

from ..model.features import cash_features

WINDOW = 6          # months (30-day buckets) before the loan date
MIN_MONTHS = 3
CASH_COLS = ["mean_inflow", "median_inflow", "inflow_cv", "dip_months", "spike_months", "weeks_of_buffer",
             "low_balance_days", "expense_to_income", "balance_level", "balance_slope", "months_observed", "emi_to_income"]


def loan_arrays(loan_row, tr):
    """tr: that account's transactions (date as datetime). Returns (inc, out, bal, last_date_used) or None."""
    ld = pd.Timestamp(loan_row["date"])
    t = tr[tr["date"] < ld]
    if t.empty:
        return None
    days = (ld - t["date"]).dt.days.values - 1               # 0 = the day before the loan
    bucket = days // 30
    t = t.assign(bucket=bucket)
    t = t[t.bucket < WINDOW]
    first_days = (ld - tr["date"].min()).days
    n_avail = min(WINDOW, int(first_days // 30))
    if n_avail < MIN_MONTHS:
        return None
    inc, out, bal = np.zeros(n_avail), np.zeros(n_avail), np.zeros(n_avail)
    t_all = tr[tr["date"] < ld].sort_values(["date", "trans_id"])
    for b in range(n_avail):                                  # bucket 0 is the most recent
        lo = ld - pd.Timedelta(days=30 * (b + 1)); hi = ld - pd.Timedelta(days=30 * b)
        seg = t_all[(t_all["date"] >= lo) & (t_all["date"] < hi)]
        inc[b] = seg.loc[seg["type"] == "PRIJEM", "amount"].sum()
        out[b] = seg.loc[seg["type"] != "PRIJEM", "amount"].sum()
        prior = t_all[t_all["date"] < hi]
        bal[b] = prior["balance"].iloc[-1] if len(prior) else 0.0
    last = t_all["date"].max()
    return inc[::-1], out[::-1], bal[::-1], last


def build(loan, trans):
    """Returns DataFrame: loan_id, account_id, y (1 = default, status B or D), cash features, date_used_max, loan_date."""
    trans = trans.assign(date=pd.to_datetime(trans["date"]))
    by_acc = {a: g for a, g in trans.groupby("account_id")}
    rows = []
    for _, lr in loan.iterrows():
        res = loan_arrays(lr, by_acc[lr["account_id"]]) if lr["account_id"] in by_acc else None
        if res is None:
            continue
        inc, out, bal, last = res
        inc, out, bal = np.maximum(inc, 1.0), np.maximum(out, 1.0), np.maximum(bal, 1.0)   # avoid zero division; amounts are positive
        f = cash_features(inc[None], out[None], bal[None], [float(lr["payments"])])
        rows.append({"loan_id": lr["loan_id"], "account_id": lr["account_id"], "y": int(lr["status"] in ("B", "D")),
                     "loan_date": pd.Timestamp(lr["date"]), "date_used_max": last, **{k: float(v[0]) for k, v in f.items()}})
    return pd.DataFrame(rows)
