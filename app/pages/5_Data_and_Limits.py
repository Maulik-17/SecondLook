import sys
sys.path.insert(0, "src")
import json

import pandas as pd
import streamlit as st

from secondlook import config as C

DISCLAIMER = "Synthetic data; an audit-tool demonstration, not a lending decision tool."
st.set_page_config(page_title="Data and limits", layout="wide")
st.title("Data and limits")


def load(name):
    p = C.ART_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


metrics, dd, berka, cal = load("metrics.json"), load("data_dictionary.json"), load("berka_sanity.json"), load("berka_calibration.json")
h = (metrics or {}).get("config_hash", "unknown")

st.subheader("Data dictionary")
if dd:
    df = pd.DataFrame(dd)
    colour = {"assumed": "background-color:#fde2c8", "literature-supported": "background-color:#d6e6f5", "berka-supported": "background-color:#d4efdc"}
    st.dataframe(df.style.map(lambda v: colour.get(v, ""), subset=["source"]), use_container_width=True)
    st.caption(f"n = {len(df)} features. Rent, utility and recharge signals are assumed to predict default; no source we read shows it.")
else:
    st.info("Data dictionary not found. Run scripts/train.py.")

st.subheader("Injected biases (switchable, on by default for the demo)")
st.markdown(
    "1. **Measurement bias:** for shared-phone and document-poor people the on-time features are noisier and biased downward (assumption).\n"
    "2. **Income-volatility gap:** women in gig occupations have higher income volatility. This is a structural pathway, not a direct effect of gender on risk.\n"
    "3. **Planted proxies:** `night_active_share` tracks gender and `local_market_share` tracks region. Neither affects true risk.\n"
    "4. **Historical selection:** a legacy rule decided who has observed outcomes; others are seen only through a small exploration share.")
st.caption("A null-world cohort with unequal base rates and no injected bias is also generated, as a control for the audit.")

st.subheader("Model metrics")
if metrics:
    rows = [{"split": s, "n": metrics[s]["n"], "AUC low": metrics[s]["auc"][0], "AUC": metrics[s]["auc"][1], "AUC high": metrics[s]["auc"][2],
             "Brier": metrics[s]["brier"], "ECE": metrics[s]["ece"]} for s in ["test", "oot"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.write(f"Out-of-time cohort has drifted income volatility and shock rates. Approval rate on validation: {metrics['approval_rate_val']:.1%}; tau = {metrics['tau']:.2f} (cost of a default set at 5 times a lost good loan, an assumption).")
    st.write("Features dropped because their fitted sign contradicted the expert direction: " + (", ".join(metrics["dropped_features"]) or "none"))
    cg = pd.DataFrame(metrics["calibration_by_gender_test"]).T.reset_index(names="gender")
    st.dataframe(cg, use_container_width=True)
else:
    st.info("Metrics not found. Run scripts/train.py.")

st.subheader("Berka sanity run")
if berka:
    lo, est, hi = berka["auc"]
    st.write(f"{berka['n']} loans, {berka['defaults']} defaults; cross-validated AUC {est:.2f} (95% interval {lo:.2f} to {hi:.2f}), "
             f"using only transactions before the loan date, {berka['window_months']}-month window.")
    st.caption(berka["note"])
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame(berka["by_gender"]), use_container_width=True)
    c2.dataframe(pd.DataFrame(berka["by_district_income_tercile"]), use_container_width=True)
    if cal:
        st.write("Calibration targets, Berka vs simulator (quantiles " + ", ".join(str(q) for q in cal["quantiles"]) + ")")
        st.dataframe(pd.DataFrame({k: {"berka": [round(x, 2) for x in v["berka"]], "simulator": [round(x, 2) for x in v["sim"]]}
                                   for k, v in cal["targets"].items()}).T, use_container_width=True)
else:
    st.info("Berka sanity file not found. Run scripts/berka_sanity.py.")

st.subheader("Claims we do not make")
st.markdown(
    "- No claim of a new metric, no promised outcomes, no claim that nothing like this exists.\n- No real credit data: everything is synthetic or Czech bank data used for calibration.\n"
    "- No claim about equalised effort across groups.\n- No caste or religion analysis.\n"
    "- Proxy AUC is not proof of discrimination, and weak proxies give weak bounds on group gaps.\n"
    "- No claim of regulatory compliance; RBI FREE-AI and DPDP texts are unread.")
st.caption(DISCLAIMER + " config_hash=" + str(h))
