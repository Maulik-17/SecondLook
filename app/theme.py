"""Shared look for every page: injects style/style.css and a bright Plotly template. Design only."""
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_CSS = Path(__file__).parent / "style" / "style.css"
_COLORWAY = ["#6D28D9", "#0EA5A4", "#FF6B57", "#2563EB", "#FFC83D", "#EC4899", "#16A34A"]

pio.templates["secondlook"] = go.layout.Template(layout=dict(
    colorway=_COLORWAY, font=dict(size=16, color="#1F2447"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
    xaxis=dict(gridcolor="#E4ECFB", linecolor="#C9D8F5", title_font_size=17, tickfont_size=15),
    yaxis=dict(gridcolor="#E4ECFB", linecolor="#C9D8F5", title_font_size=17, tickfont_size=15),
    legend=dict(font_size=15), title_font_size=20))
pio.templates.default = "secondlook"


def apply():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)
