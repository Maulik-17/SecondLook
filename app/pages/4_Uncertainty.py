import json
import sys

sys.path.insert(0, "src")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from secondlook import config as C
from secondlook.audit.data import DISCLAIMER

st.set_page_config(page_title="Uncertainty", layout="wide")
st.title("Uncertainty and referral")
st.warning("Audit-tool demonstration on synthetic data. Not findings about real lenders.")

eps = st.slider("Exploration rate eps", 0.01, 0.10, float(C.EXPLORATION_EPS), 0.01)
alpha = st.slider("Miscoverage alpha", 0.05, 0.20, float(C.ALPHA), 0.01)
if st.button("Recompute"):
    from secondlook.uncertainty.conformal import run_uncertainty
    with st.spinner("Recomputing"):
        st.session_state["unc"] = run_uncertainty(eps, alpha)

U = st.session_state.get("unc")
if U is None:
    f = C.ART_DIR / "uncertainty.json"
    if not f.exists():
        st.error("uncertainty.json is missing. Regenerate it with: python scripts/run_audit.py")
        st.stop()
    U = json.load(open(f))
st.write(f"Showing eps = {U['eps']}, alpha = {U['alpha']}.")
for w in U.get("warnings", []):
    st.warning(w)

V = U["variants"]["weighted"]
attrs = list(dict.fromkeys(g["attribute"] for g in V["groups"])) if V.get("groups") else []
if V.get("status") == "ok":
    st.header("Weighted calibration (uses exploration weights)")
    a = st.selectbox("Attribute", attrs)
    g = pd.DataFrame([x for x in V["groups"] if x["attribute"] == a])
    c1, c2 = st.columns(2)
    for col, key, title, tgt in ((c1, "coverage", "Coverage by group", 1 - U["alpha"]), (c2, "referral", "Referral (abstain) by group", None)):
        fig = go.Figure(go.Bar(x=g["level"].astype(str), y=g[key], text=[f"n={n}" for n in g["n"]]))
        if tgt:
            fig.add_hline(y=tgt, line_dash="dash")
        fig.update_layout(title=title, height=320)
        col.plotly_chart(fig, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Coverage gap (max over attributes)", f"{V['coverage_gap']:.3f}")
    m2.metric("Referral gap (max over attributes)", f"{V['referral_gap']:.3f}")
    m3.metric("Overall coverage / referral", f"{V['overall_coverage']:.3f} / {V['overall_referral']:.3f}")
    st.caption("Exact group-wise coverage and equal set size cannot both hold (Gao et al.), so both gaps are shown together.")

st.header("Three calibration variants")
for name, v in U["variants"].items():
    st.subheader(name.replace("_", " "))
    if v.get("status") != "ok":
        st.info(f"{v.get('status')} (n_cal={v.get('n_cal')})")
        continue
    t = pd.DataFrame(v["groups"])[["attribute", "level", "n", "coverage", "referral", "ess"]].round(3)
    st.write(f"calibration rows: {v['n_cal']}; coverage gap {v['coverage_gap']:.3f}; referral gap {v['referral_gap']:.3f}; overall coverage {v['overall_coverage']:.3f}")
    st.dataframe(t, use_container_width=True)

st.header("Pool vs fresh population")
pf = C.ART_DIR / "poolfresh.json"
if pf.exists():
    P = json.load(open(pf))
    series = P.get("series", {"selected": P})
    key = st.selectbox("Exploration rate for the stream", list(series), index=min(1, len(series) - 1))
    s = series[key]
    fig = go.Figure([go.Scatter(x=s["round"], y=s["pool_recall"], name="pool (observed applicants)"),
                     go.Scatter(x=s["round"], y=s["fresh_recall"], name="fresh population (test)")])
    fig.update_layout(xaxis_title="round", yaxis_title="share of defaulters the refitted model rejects", height=330)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Monitoring on the pool can look healthy while screening on the real population does not. The plot shows one simulated stream; do not read more into it.")
else:
    st.info("poolfresh.json is missing. Regenerate it with: python scripts/run_audit.py")

st.header("Assumptions")
st.markdown("""
- Outcomes are only seen for approved people and a small random sample of rejects; the sampling probability is known by design.
- Reweighting by 1/propensity assumes the outcome is independent of selection given the features, and that every rejected person has a nonzero chance of being explored. Weights are capped at W_MAX, which adds bias.
- Calibration is split into cells (thin file by gender); small cells have a low effective sample size.
- Applicants and loan officers may not understand conformal sets; this is an engineering demonstration.
""")
st.caption(f"{DISCLAIMER}  config_hash={U.get('config_hash', 'see audit page')}")
