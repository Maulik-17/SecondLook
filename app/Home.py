import sys
sys.path.insert(0, "src")
import streamlit as st

from secondlook import config as C

DISCLAIMER = "Synthetic data; an audit-tool demonstration, not a lending decision tool."
st.set_page_config(page_title="SecondLook", layout="wide")

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme
theme.apply()

st.title("SecondLook")
st.write(
    "SecondLook scores people with thin credit files, such as gig workers, shopkeepers and students, from cash-flow signals, "
    "then audits the score. It asks who carries the burden of fixing a rejection, how sure the score is on thin files "
    "(unsure cases go to a person), which repayment plan still works when income is volatile, and what can be said when "
    "group labels are missing.")
with st.container(border=True):
    st.subheader("Real vs simulated vs assumed")
    st.markdown(
        "- **Real:** Berka, a Czech bank dataset from the 1990s, is used for a sanity run and to calibrate income variation. "
        "It is not Indian data.\n"
        "- **Simulated:** every person, account history and loan outcome in the demo.\n"
        "- **Assumed:** rent, utility and recharge on-time signals, and the two planted proxy features. "
        "No source we read shows they predict default.")
st.page_link("pages/1_Applicant.py", label="Applicant: score one person and see reasons")
for label, f in [("Audit", "pages/2_Audit.py"), ("Recourse", "pages/3_Recourse.py"), ("Uncertainty", "pages/4_Uncertainty.py")]:
    import os
    if os.path.exists(os.path.join("app", f)):
        st.page_link(f, label=label)
st.page_link("pages/5_Data_and_Limits.py", label="Data and limits: dictionary, injected biases, metrics")
st.caption(DISCLAIMER)
