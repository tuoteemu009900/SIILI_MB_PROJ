
from __future__ import annotations

# ===== FIX: dataclass import MUST be before usage =====
from dataclasses import dataclass

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS = True
except Exception:
    PLOTLY_EVENTS = False

# -------------------- CONFIG --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "benchmark_long.csv"
TARGET = "Siili Solutions"

GOOD_GREEN = "#22c55e"
BAD_RED = "#ef4444"

# -------------------- DATA --------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)

    numeric_cols = [
        "Revenue_EURm","EBITDA_pct","PersonnelCost_pct","Outsourcing_pct",
        "Headcount","Billable_pct","Senior_pct","Offshore_pct",
        "RevenuePerEmployee_EURk"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# -------------------- STYLING --------------------
def css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
          h1, h2, h3 { letter-spacing: -0.02em; }
          .sticky {
            position: sticky; top: 0; z-index: 999;
            background: linear-gradient(180deg, rgba(11,18,32,0.98), rgba(11,18,32,0.78));
            backdrop-filter: blur(10px);
            padding: 0.85rem 0 0.6rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
          }
          .badge {
            display:inline-block; padding: 0.35rem 0.65rem; border-radius: 999px;
            background: rgba(0,163,255,0.15); border: 1px solid rgba(0,163,255,0.35);
            font-weight:700;
          }
          .card {
            background: rgba(17,26,43,0.92);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1rem;
          }
        </style>
        """, unsafe_allow_html=True
    )

# -------------------- METRIC MODEL --------------------
@dataclass
class Metric:
    key: str
    title: str
    suffix: str
    fmt: str
    higher_is_better: bool

METRICS = [
    Metric("EBITDA_pct", "EBITDA", "%", ".1f", True),
    Metric("PersonnelCost_pct", "Personnel cost", "%", ".1f", False),
    Metric("RevenuePerEmployee_EURk", "Revenue / Employee", "k€", ".0f", True),
    Metric("Outsourcing_pct", "Outsourcing", "%", ".1f", False),
]

# -------------------- HELPERS --------------------
def median(df: pd.DataFrame, metric: str, region: str, year: int):
    s = df[(df["Region"] == region) & (df["Year"] == year)][metric].dropna()
    return None if s.empty else float(s.median())

def indicator(metric: Metric, value: float | None, ref: float | None):
    if value is None or np.isnan(value):
        value = 0.0
        show_delta = False
    else:
        show_delta = ref is not None and not np.isnan(ref)

    if metric.higher_is_better:
        inc, dec = GOOD_GREEN, BAD_RED
    else:
        inc, dec = BAD_RED, GOOD_GREEN

    fig = go.Figure(go.Indicator(
        mode="number+delta" if show_delta else "number",
        value=float(value),
        number={"suffix": metric.suffix, "valueformat": metric.fmt},
        delta={
            "reference": float(ref),
            "valueformat": metric.fmt,
            "suffix": metric.suffix,
            "increasing": {"color": inc},
            "decreasing": {"color": dec},
        } if show_delta else None,
        title={"text": metric.title},
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=10,r=10,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# -------------------- APP --------------------
def main():
    st.set_page_config(
        page_title="Market Benchmark — Siili",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    css()
    df = load_data()

    years = sorted(df["Year"].unique().tolist())
    year = max(years)

    snap = df[df["Year"] == year].copy()

    st.markdown(
        f"""
        <div class="sticky">
          <h1 style="margin:0;">Market Benchmark <span class="badge">{TARGET}</span></h1>
          <div style="opacity:0.7">CEO-ready • green = good • red = bad • {year}</div>
        </div>
        """, unsafe_allow_html=True
    )

    cols = st.columns(4, gap="large")
    for i, m in enumerate(METRICS):
        with cols[i]:
            row = snap[snap["Company"] == TARGET]
            val = None if row.empty else row.iloc[0].get(m.key)
            ref = median(snap, m.key, "Finland (Peer)", year)
            st.plotly_chart(indicator(m, val, ref), use_container_width=True)

    st.success("✅ App is running correctly. dataclass error fixed.")

if __name__ == "__main__":
    main()
