import sys
sys.path.insert(0, "src")
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from secondlook import config as C
from secondlook.contracts import FEATURES
from secondlook.model import explain
from secondlook.model.scorer import load_model

DISCLAIMER = "Synthetic data; an audit-tool demonstration, not a lending decision tool."
st.set_page_config(page_title="Applicant", layout="wide")
st.title("Applicant")


@st.cache_resource
def load():
    s = load_model()
    F = pd.read_parquet(C.DATA_DIR / "features.parquet")
    P = pd.read_parquet(C.DATA_DIR / "people.parquet")
    return s, F, P


s, F, P = load()
h = getattr(s, "config_hash", "unknown")

# ---- consent
with st.container(border=True):
    st.subheader("Consent")
    st.markdown(
        "- **Data used:** summaries of income, spending and balance from the account history shown here. No contacts, location or group attributes.\n"
        "- **How long:** this browser session only. Nothing is stored on a server.\n"
        "- **Purpose:** to demonstrate scoring and its explanation on a synthetic person.")
    if st.button("Revoke consent"):
        st.session_state.clear()
        st.warning("Consent revoked and session data cleared.")
        st.stop()
st.session_state.setdefault("ledger", [])

# ---- choose person
meta = P.set_index("id")
ids = P[P.split != "oot"].id.tolist()[:500]
pid = st.selectbox("Synthetic person", ids, format_func=lambda i: f"id {i} | {meta.loc[i, 'occupation']} | thin_file={meta.loc[i, 'thin_file']}")
x = F[F.id == pid].copy()
with st.expander("Edit five key features"):
    for f, lo, hi in [("weeks_of_buffer", 0.0, 30.0), ("inflow_cv", 0.0, 1.5), ("dip_months", 0.0, 12.0),
                      ("emi_to_income", 0.0, 1.0), ("expense_to_income", 0.2, 1.5)]:
        v = float(x[f].iloc[0])
        x[f] = st.slider(f, lo, hi, float(min(max(v, lo), hi)))

p = float(s.predict_proba(x)[0])
tau = st.slider("Decision threshold tau (default is the cost-based value)", 0.01, 0.99, float(s.tau), 0.01)
conf = None
try:
    from secondlook.uncertainty.conformal import load_conformal
    g = P[P.id == pid][["gender", "urban_rural", "region", "occupation", "thin_file", "shared_phone", "document_poor"]]
    conf = load_conformal().predict_set(np.array([p]), g)[0]
except Exception:
    conf = None
refer = bool(conf and conf.get("abstain"))
decision = "Refer to a person" if refer else ("Approve" if p <= tau else "Decline")
if st.session_state.get("last") != (pid, tau, p):
    st.session_state["ledger"].append({"time": datetime.now().strftime("%H:%M:%S"), "purpose": "score synthetic person " + str(pid),
                                       "data used": "cash-flow summary features"})
    st.session_state["last"] = (pid, tau, p)

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Decision", decision)
    st.metric("Estimated chance of default", f"{p:.1%}")
    st.caption(f"threshold tau = {tau:.2f}; thin file: {'yes' if meta.loc[pid, 'thin_file'] else 'no'}")
    if refer:
        st.info("We are not sure about this application: referred to a person.")
with c2:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number={"suffix": "%"},
                                 gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"width": 3}, "value": 100 * tau}}))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
st.subheader("Top reasons")
for r in explain.reasons(s, x):
    st.markdown(f"- {r['text']} ({r['direction']}, {r['contribution']:+.2f})")
st.subheader("Data-use ledger")
st.dataframe(pd.DataFrame(st.session_state["ledger"]), use_container_width=True)
st.caption(DISCLAIMER + " config_hash=" + str(h))
