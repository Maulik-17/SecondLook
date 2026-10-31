import json
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from secondlook import config as C
from secondlook.audit.nolabel import nolabel_bounds

st.set_page_config(page_title="Audit", layout="wide")
st.title("Audit")
st.warning("Audit-tool demonstration on synthetic data. Not findings about real lenders.")

f = C.ART_DIR / "audit_summary.json"
if not f.exists():
    st.error("audit_summary.json is missing. Regenerate it with: python scripts/run_audit.py")
    st.stop()
A = json.load(open(f))


def ci(x, d=3):
    return f"{x[1]:.{d}f} [{x[0]:.{d}f}, {x[2]:.{d}f}]" if x and x[1] is not None and not np.isnan(x[1]) else "n/a"


attrs = sorted({g["attribute"] for g in A["groups"]}, key=[g["attribute"] for g in A["groups"]].index)
attr = st.selectbox("Attribute", attrs)
min_n = C.MIN_GROUP_N

# 2. group table
st.header("Group table")
st.caption("Outcome metrics (PPV, TPR, FPR) are among the approved / by approval status; evaluation population is the test split.")
rows = [g for g in A["groups"] if g["attribute"] == attr]
tbl = pd.DataFrame([{"level": g["level"], "n": g["n"], "approval": ci(g["approval"]), "base rate": ci(g["base_rate"]),
                     "PPV (among approved)": ci(g["ppv"]), "TPR": ci(g["tpr"]), "FPR": ci(g["fpr"]),
                     "ECE": f"{g['ece']:.3f}", "Brier": f"{g['brier']:.3f}",
                     "missing outcome share": f"{g['missing_outcome_share']:.3f}", "n >= 50": g["n"] >= min_n} for g in rows])
st.dataframe(tbl.style.apply(lambda r: ["color: #999" if not r["n >= 50"] else "" for _ in r], axis=1), use_container_width=True)
di = A.get("disparate_impact", {}).get(attr)
st.metric("Approval-rate ratio (min / max across levels with n >= 50)", "n/a" if di is None else f"{di:.2f}")
st.caption(A.get("disparate_impact_note", ""))

# 3. burden
st.header("Burden on the creditworthy who were rejected")
st.caption(A.get("burden_note", ""))
b = [r for r in A["burden"] if r["attribute"] == attr]
worst = A["burden_headline"]["worst"]
if worst:
    r_ = A["burden_headline"]["ratio"]
    st.metric("Worst group (all attributes)", f"{worst['attribute']}: {worst['level']}", f"worst/best = {r_:.2f}" if r_ else None, delta_color="off")
colors = ["#d62728" if (worst and r["attribute"] == worst["attribute"] and r["level"] == worst["level"]) else "#4c78a8" for r in b]
fig = go.Figure(go.Bar(x=[str(r["level"]) for r in b], y=[r["burden"][1] for r in b], marker_color=colors,
                       error_y=dict(type="data", symmetric=False, array=[r["burden"][2] - r["burden"][1] for r in b],
                                    arrayminus=[r["burden"][1] - r["burden"][0] for r in b]),
                       text=[f"n={r['n']}" for r in b]))
fig.update_layout(yaxis_title="mean effort per group member (rejected and creditworthy)", height=350)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(pd.DataFrame([{"level": r["level"], "n": r["n"], "rejected and creditworthy": r["n_rejected_good"],
                            "burden": ci(r["burden"]), "infeasible share": round(r["infeasible_share"], 3)} for r in b]), use_container_width=True)

# 4. budget curves
st.header("Effort-budget curves")
bc = [c for c in A["budget_curve"] if c["attribute"] == attr]
fig = go.Figure([go.Scatter(x=c["budget"], y=c["share"], name=f"{c['level']} (n={c['n']})") for c in bc])
fig.update_layout(xaxis_title="effort budget", yaxis_title="share of rejected creditworthy who can reach approval", height=330)
st.plotly_chart(fig, use_container_width=True)

# 5. risk-matched gap
st.header("Risk-matched gap")
st.caption("Gap in mean effort between the highest-burden level and the rest, within quintiles of true risk (rejected people). "
           "A gap near zero means the raw difference is explained by risk.")
rm = [r for r in A["risk_matched_gap"] if r["attribute"] == attr]
if rm:
    fig = go.Figure(go.Bar(x=[f"Q{r['bin'] + 1} (n={r['n']})" for r in rm], y=[r["gap"][1] for r in rm],
                           error_y=dict(type="data", symmetric=False, array=[r["gap"][2] - r["gap"][1] for r in rm],
                                        arrayminus=[r["gap"][1] - r["gap"][0] for r in rm])))
    fig.update_layout(height=300, yaxis_title=f"gap, level '{rm[0]['level']}' minus rest")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough rejected people to match on risk.")

# 6. level-down
st.header("Level-down")
ld = A["level_down"]
tau_sel = st.slider("Threshold tau", float(min(x["tau"] for x in ld)), float(max(x["tau"] for x in ld)), float(A["tau"]), 0.01)
near = min(ld, key=lambda x: abs(x["tau"] - tau_sel))
fig = go.Figure(go.Scatter(x=[x["approval_rate"] for x in ld], y=[x["worst_group_burden"] for x in ld], mode="lines+markers",
                           text=[f"tau={x['tau']:.2f}" for x in ld], name="curve"))
fig.add_trace(go.Scatter(x=[near["approval_rate"]], y=[near["worst_group_burden"]], mode="markers",
                         marker=dict(size=16, color="#d62728"), name=f"tau={near['tau']:.2f}"))
fig.update_layout(xaxis_title="approval rate (test split)", yaxis_title="worst-group burden", height=330)
st.plotly_chart(fig, use_container_width=True)
if A["failure_flags"]["level_down"]:
    st.warning("Level-down: a stricter threshold lowers the burden ratio by rejecting more people. "
               "A gap that shrinks because approvals fall is not a fairer system.")

# 7. failure cards
st.header("Failure-mode cards")
fp = A["failure_panel"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Level-down", "flagged" if fp["level_down"] else "quiet")
c2.metric("Inversion", "flagged" if fp["inversion"] else "quiet", f"{fp['inversion_count']} people", delta_color="off")
c3.metric("Gaming", "flagged" if fp["gaming"] else "quiet", f"ratio {fp['gaming_ratio']:.2f}", delta_color="off")
c4.metric("Infeasible share", f"{fp['infeasible_share']:.3f}", f"p95 effort {fp['effort_p95']:.2f}" if fp.get("effort_p95") is not None else None, delta_color="off")
st.caption(fp["gaming_note"] + " " + fp["extraction_note"])
if fp.get("referral_view"):
    st.subheader("Referral view (rejected or referred to a person), creditworthy people")
    rv = pd.DataFrame([r for r in fp["referral_view"] if r["attribute"] == attr])
    st.dataframe(rv.rename(columns={"n_good": "n (creditworthy)"}), use_container_width=True)
else:
    st.caption("Referral view needs a saved conformal model (run scripts/run_audit.py).")
if A.get("null_world_quiet") is not None:
    st.caption(f"Null-world check (unequal base rates, no injected bias): {'quiet, as it should be' if A['null_world_quiet'] else 'NOT quiet'}.")

# 8. weight sensitivity
st.header("Weight sensitivity")
ws = A["weight_sensitivity"]
st.dataframe(pd.DataFrame([{"weight set": k, "worst group": f"{v['worst']['attribute']}: {v['worst']['level']}" if v["worst"] else "n/a",
                            "worst/best": round(v["ratio"], 2) if v["ratio"] else None} for k, v in ws["by_weight_set"].items()]))
st.write("Worst group and gap signs stable across weight sets:", ws["sign_stable"])
st.caption("For a linear scorecard the weight sets rescale all efforts together unless a feature range limit binds, so little change is expected.")

# 9. proxy leakage
st.header("Proxy leakage")
st.dataframe(pd.DataFrame([{"attribute": p["attribute"], "n": p["n"], "AUC (logistic)": ci(p["auc"]), "AUC (boosted)": ci(p["auc_boosted"]),
                            "permutation null 95%": round(p["permutation_null_95"], 3), "p-value": round(p["p_value"], 3),
                            "planted proxy is top feature": p["planted_top"], "equal-risk approval gap": ci(p["conditional_gap"])}
                           for p in A["proxy_leakage"]]), use_container_width=True)
st.info("Reconstructible is not the same as discriminatory: a high AUC with a small equal-risk gap means the features reveal the attribute but approval does not differ at equal risk.")

# 10. no-label mode
st.header("No-label mode")
st.caption("Group labels are missing; only a proxy cell (region) and an external table P(A=1 | cell) are known. Only an interval is shown. "
           "Assumption: the interval is the identified set under the observed approval-by-cell and the external table (partial identification in the spirit of Kallus, Mao and Zhou, arXiv 1906.00285; not checked against their estimator).")
nl = A.get("nolabel", [])
if nl:
    pick = st.selectbox("Hidden attribute", [f"{x['attribute']} ({x['level1']} minus rest)" for x in nl])
    x = nl[[f"{y['attribute']} ({y['level1']} minus rest)" for y in nl].index(pick)]
    st.write(f"Disparity in approval rate lies in [{x['lower']:.3f}, {x['upper']:.3f}] (width {x['width']:.3f}, n={x['n']}).")
up = st.file_uploader("Or upload a CSV with columns cell, p_a1, share, approval_rate", type="csv")
if up is not None:
    u = pd.read_csv(up)
    if {"cell", "p_a1", "share", "approval_rate"} <= set(u.columns):
        r = nolabel_bounds(u[["cell", "p_a1", "share"]], u[["cell", "approval_rate"]])
        st.write(f"Interval: [{r['lower']:.3f}, {r['upper']:.3f}] from {len(u)} cells.")
    else:
        st.error("Missing columns.")

# 11. trade-off
st.header("Trade-off")
st.write(A["tradeoff"]["note"])
td = pd.DataFrame([r for r in A["tradeoff"].get("rows", []) if r["attribute"] == attr])
if len(td):
    st.dataframe(td.round(3), use_container_width=True)

# 12. months to approval
st.header("Months to approval")
mt = A.get("months_to_approval")
if mt:
    st.dataframe(pd.DataFrame([r for r in mt if r["attribute"] == attr]), use_container_width=True)
else:
    st.info("Months-to-approval is not available yet (recourse results not generated).")

st.caption(f"{A['disclaimer']}  config_hash={A['config_hash']}")
