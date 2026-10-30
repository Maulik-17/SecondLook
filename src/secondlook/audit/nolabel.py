"""Bounds on a group disparity in approval when the group label is missing, given a proxy cell per person and an
external table P(A=1 | cell).

Assumption stated plainly: this is the sharp identified set under exactly two sets of constraints, (i) the observed
joint of (cell, approval) and (ii) the externally known P(A | cell). It is the partial-identification idea of
Kallus, Mao and Zhou (arXiv 1906.00285) written as a small linear program; it was not checked line by line against
the paper's own estimator, so it should not be described as their method."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def nolabel_bounds(proxy_table, approval_by_cell):
    """proxy_table: columns cell, p_a1 (P(A=1|cell)), share (P(cell)).
    approval_by_cell: columns cell, approval_rate. Returns {"lower","upper","width"} for P(Y=1|A=1)-P(Y=1|A=0)."""
    t = proxy_table.merge(approval_by_cell, on="cell")
    t = t.assign(share=t.share / t.share.sum())
    Z = len(t)
    pz, pa1, ap = t.share.values, t.p_a1.values, t.approval_rate.values
    # variable order: a[z, y, g] -> index z*4 + y*2 + g
    ix = lambda z, y, g: z * 4 + y * 2 + g
    A_eq, b_eq = [], []
    for z in range(Z):
        for y in (0, 1):                                   # sum_g a[z,y,g] = P(z, y)
            row = np.zeros(4 * Z)
            row[ix(z, y, 0)] = row[ix(z, y, 1)] = 1
            A_eq.append(row)
            b_eq.append(pz[z] * (ap[z] if y == 1 else 1 - ap[z]))
        for g in (0, 1):                                   # sum_y a[z,y,g] = P(z) P(g|z)
            row = np.zeros(4 * Z)
            row[ix(z, 0, g)] = row[ix(z, 1, g)] = 1
            A_eq.append(row)
            b_eq.append(pz[z] * (pa1[z] if g == 1 else 1 - pa1[z]))
    PA1 = float((pz * pa1).sum())
    PA0 = 1 - PA1
    if PA1 <= 0 or PA0 <= 0:
        return {"lower": float("nan"), "upper": float("nan"), "width": float("nan")}
    c = np.zeros(4 * Z)
    for z in range(Z):
        c[ix(z, 1, 1)] = 1 / PA1
        c[ix(z, 1, 0)] = -1 / PA0
    lo = linprog(c, A_eq=np.array(A_eq), b_eq=b_eq, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=np.array(A_eq), b_eq=b_eq, bounds=(0, None), method="highs")
    lower, upper = float(lo.fun), float(-hi.fun)
    return {"lower": lower, "upper": upper, "width": upper - lower}


def tables_from_data(df, attr, level1, tau, cell="region"):
    """Proxy table (from the population) and approval-by-cell (from decisions) for a chosen attribute."""
    g = df.groupby(cell)
    prox = pd.DataFrame({"cell": g.size().index, "p_a1": g.apply(lambda d: float((d[attr] == level1).mean())).values,
                         "share": (g.size() / len(df)).values})
    appr = pd.DataFrame({"cell": g.size().index, "approval_rate": g.apply(lambda d: float((d.p_default <= tau).mean())).values})
    return prox, appr


def true_disparity(df, attr, level1, tau):
    a = df[attr] == level1
    ap = df.p_default <= tau
    return float(ap[a].mean() - ap[~a].mean())


def nolabel_summary(df, tau):
    out = []
    for attr, lv in (("gender", "F"), ("urban_rural", "rural")):
        prox, appr = tables_from_data(df, attr, lv, tau)
        b = nolabel_bounds(prox, appr)
        out.append({"attribute": attr, "level1": lv, "proxy": "region", "n": int(len(df)), **b,
                    "true_disparity": true_disparity(df, attr, lv, tau)})
    return out
