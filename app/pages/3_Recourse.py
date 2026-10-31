import sys
sys.path.insert(0, "src")

import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from secondlook import config as C
from secondlook.recourse import actions as A
from secondlook.recourse.context import get_ctx
from secondlook.recourse.plan import plan as _plan

DISCLAIMER = "Audit tool demonstration, not findings about real lenders. Data are synthetic (Berka-calibrated)."

st.set_page_config(page_title="Recourse", layout="wide")
st.title("Recourse: cheapest plan under volatile income")
st.caption(DISCLAIMER)


@st.cache_data(show_spinner="Simulating income paths...")
def cached_plan(pid, horizon, target, rho, tau):
    return _plan(pid, horizon=horizon, target_prob=target, rho_override=rho, tau=tau)


try:
    ctx = get_ctx()
except Exception as ex:  # missing data or model
    st.error(f"Data or model not found ({ex}). Set SL_DATA and SL_ART or run the data and training scripts.")
    st.stop()

tau0 = float(ctx["scorer"].tau)
cfg_hash = hashlib.sha256(repr({k: getattr(C, k) for k in dir(C) if k.isupper() and k not in ("ROOT", "DATA_DIR", "ART_DIR")}).encode()).hexdigest()[:10]
st.caption(f"config_hash: {cfg_hash} | approval threshold tau = {tau0:.3f}")

people, scores = ctx["people"], ctx["scores"]
rej = people[(people.split == "test")].join(scores[["p_default", "decision"]])
rej = rej[rej.decision == 0].sort_index()
if rej.empty:
    st.warning("No rejected test-split people in this dataset.")
    st.stop()
labels = {int(i): f"id {i} | {r.occupation} | thin_file={int(r.thin_file)} | p_default={r.p_default:.3f}" for i, r in rej.head(500).iterrows()}
pid = st.selectbox("Rejected applicant", list(labels), format_func=labels.get)

c1, c2, c3 = st.columns(3)
horizon = c1.slider("Horizon (months)", 1, 24, 12)
target = c2.slider("Target probability", 0.5, 0.95, 0.8, 0.05)
use_fit = c3.checkbox("Use fitted persistence", True)
rho = None if use_fit else c3.slider("Persistence rho", 0.0, 0.9, 0.45, 0.05)

if st.button("Find cheapest plan") or st.session_state.get("ran"):
    st.session_state["ran"] = True
    r = cached_plan(int(pid), horizon, target, rho, None)
    lo, est, hi = r["p_success"]
    q = r["months_to_approval_quantiles"]
    if r.get("infeasible"):
        st.warning("The target probability cannot be reached with the available actions; showing the strongest plan.")
    if r["fragile"]:
        st.markdown(":red-background[FRAGILE: meets the target in the base case but drops below 0.6 under at least one stress test]")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total cost (effort points, assumed)", f"{r['cost']:.1f}")
    m2.metric("P(approved within horizon)", f"{est:.2f}", f"95% interval {lo:.2f} to {hi:.2f}", delta_color="off")
    m3.metric("Months to approval p10 / p50 / p90", " / ".join("never" if q[k] is None else f"{q[k]:.0f}" for k in ("p10", "p50", "p90")))
    if r["plan"]:
        st.dataframe(pd.DataFrame(r["plan"]).rename(columns={"amount": "amount (share)", "cost": "cost (assumed)"}), hide_index=True)
    else:
        st.info("No change needed: the applicant is already approved or approval is expected without action.")

    fc = r["fan_chart"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["month"], y=fc["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["month"], y=fc["q10"], fill="tonexty", line=dict(width=0), name="10th to 90th percentile"))
    fig.add_trace(go.Scatter(x=fc["month"], y=fc["q50"], name="median projected p_default"))
    fig.add_hline(y=r.get("tau", tau0), line_dash="dash", annotation_text="tau")
    fig.update_layout(xaxis_title="Month", yaxis_title="Projected p_default", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stress tests (success probability of this plan)")
    desc = {"drift": "income volatility 20% higher", "threshold_shift": "lender threshold tau lowered by 0.02",
            "gaming": "gain from assumed rent/utility/recharge signals removed", "persistence": "worst case over rho = 0, 0.45, 0.9",
            "retrain": "lender retrains the model on drifted data (5 refits, each with its own threshold) before you finish"}
    st.dataframe(pd.DataFrame([{"test": k, "what changes": desc[k], "P(success)": round(v, 3)} for k, v in r["stress"].items()]), hide_index=True)

with st.expander("Action costs and levels (assumptions, not estimates)"):
    st.dataframe(pd.DataFrame([{"action": a, "levels": d["levels"], "cost per month (one-off for smaller_loan)": d["cost"]} for a, d in A.ACTIONS.items()]), hide_index=True)

st.subheader("Months to approval by group (moderate fixed plan, rejected test people)")
f = C.ART_DIR / "recourse_by_group.json"
if f.exists():
    g = pd.DataFrame(json.loads(f.read_text())["groups"])
    g["label"] = g.attribute + "=" + g.level.astype(str) + " (n=" + g.n.astype(str) + ")"
    st.plotly_chart(go.Figure(go.Bar(x=g.label, y=g.median_months)).update_layout(yaxis_title="median months (p50)", height=360), use_container_width=True)
    st.dataframe(g[["attribute", "level", "n", "median_months", "never_share"]], hide_index=True)
else:
    st.info("recourse_by_group.json not found. Run scripts/run_recourse.py.")

st.divider()
st.caption(DISCLAIMER + " This is a model-based probability on synthetic data, not a guarantee. "
           "The rent, utility and recharge signals are assumed. "
           "This tool reveals how the score reacts to behaviour; real deployments must consider misuse.")
