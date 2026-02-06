
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Market Benchmark – Siili Solutions (Public)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOOD_GREEN = "#2ED573"
BAD_RED = "#FF4757"
NEUTRAL = "#9AA4B2"

def delta_color(delta: float, good_if_higher: bool) -> str:
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return NEUTRAL
    good = delta >= 0 if good_if_higher else delta <= 0
    return GOOD_GREEN if good else BAD_RED

def fmt_pct(x: float, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}%"

def fmt_meur(x: float, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}€m"

def fmt_keur(x: float, d: int = 0) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}k€"

TARGET = "Siili Solutions"

@st.cache_data(show_spinner=False)
def load_public():
    snap = pd.read_csv("data/benchmark_snapshot_public.csv")
    trends = pd.read_csv("data/benchmark_trends_public.csv")
    return snap, trends

df_snap_all, df_trends_all = load_public()

# Sidebar
st.sidebar.markdown("## Controls")
years = sorted(pd.to_numeric(df_snap_all["Year"], errors="coerce").dropna().astype(int).unique().tolist())
if len(years) == 0:
    st.sidebar.error("No valid 'Year' values found in data/benchmark_snapshot_public.csv")
    st.stop()
snapshot_year = years[-1] if len(years) == 1 else int(st.sidebar.select_slider("Snapshot year", options=years, value=years[-1]))

trend_metric = st.sidebar.selectbox(
    "Trend metric",
    options=["Revenue (€m)", "EBITDA (%)", "Headcount", "Revenue / Employee (k€)", "EBITDA (€m)"],
    index=1,
)

# Optional scenario sliders (model-based, not data)
st.sidebar.markdown("---")
st.sidebar.markdown("## Scenario (what-if) – EBITA impact")
util_delta = st.sidebar.slider("Utilization change (pp)", -10.0, 10.0, 0.0, 0.5)
rate_delta = st.sidebar.slider("Billing rate change (%)", -10.0, 10.0, 0.0, 0.5)
wage_infl = st.sidebar.slider("Wage inflation (%)", 0.0, 10.0, 3.0, 0.5)

# Data for selected year
df_snap = df_snap_all[df_snap_all["Year"] == snapshot_year].copy()
df_trends = df_trends_all.copy()

# Header
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
      <div style="font-size:46px;font-weight:800;letter-spacing:-0.5px;">Market Benchmark</div>
      <div style="font-size:22px;font-weight:700;padding:10px 16px;border-radius:999px;background:rgba(27,163,255,0.18);border:1px solid rgba(27,163,255,0.35);">
        Siili Solutions (Public)
      </div>
    </div>
    <div style="opacity:0.8;margin-top:-8px;">
      Public-only • no internal/estimated/inferred data • snapshot year: <b>{snapshot_year}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI cards (Siili only in v7 public dataset)
row = df_snap.iloc[0] if len(df_snap) else None
kpi_cols = st.columns(4)

kpis = [
    ("EBITDA (%)", True, "pct"),
    ("Revenue (€m)", True, "meur"),
    ("Revenue / Employee (k€)", True, "keur"),
    ("Headcount", True, "int"),
]

for i, (key, good_if_higher, typ) in enumerate(kpis):
    with kpi_cols[i]:
        val = row[key] if row is not None and key in row else np.nan
        if typ == "pct":
            main = fmt_pct(val, 1)
        elif typ == "meur":
            main = fmt_meur(val, 1)
        elif typ == "keur":
            main = fmt_keur(val, 0)
        else:
            main = "–" if pd.isna(val) else f"{int(val)}"
        st.markdown(
            f"""
            <div style="padding:18px;border-radius:18px;background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.08);box-shadow:0 8px 30px rgba(0,0,0,0.35);">
              <div style="font-size:14px;opacity:0.85;">{key}</div>
              <div style="font-size:44px;font-weight:800;line-height:1;margin-top:10px;">{main}</div>
              <div style="opacity:0.65;font-size:12px;margin-top:10px;">
                Confidence: <b>{row['Data confidence'] if row is not None else '–'}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Executive insights
left, right = st.columns([1.25, 1.0], gap="large")
with left:
    st.markdown("### Executive insights")
    if row is None:
        st.warning("No snapshot row found.")
    else:
        st.markdown(f"- **Revenue 2024:** {fmt_meur(row['Revenue (€m)'])}")
        st.markdown(f"- **EBITDA margin 2024:** {fmt_pct(row['EBITDA (%)'])}")
        st.markdown(f"- **Employees (end of year):** {int(row['Headcount'])}")
        st.markdown(f"- **Revenue / employee:** {fmt_keur(row['Revenue / Employee (k€)'])}")
        st.caption(f"Source: {row['Source']}")

with right:
    st.markdown("### Trends")
    d = df_trends.sort_values("Year").copy()
    fig = px.line(d, x="Year", y=trend_metric, markers=True)
    fig.update_layout(template="plotly_dark", height=330, margin=dict(l=10,r=10,t=10,b=10), yaxis_title=trend_metric)
    st.plotly_chart(fig, use_container_width=True)

# Tabs
tabs = st.tabs(["Snapshot table", "Scenario (what-if)", "Data & Export"])

with tabs[0]:
    st.markdown("### Snapshot table")
    st.dataframe(df_snap, use_container_width=True, hide_index=True)
    st.info("To benchmark peers with public data, add peer rows to the CSV with the same column names. This public-only version will automatically include them.")

with tabs[1]:
    st.markdown("### Scenario (what-if) – EBITA impact (directional)")
    st.caption("This is a simple model for executive discussion. It does **not** use internal data beyond the public snapshot values.")
    if row is None or pd.isna(row["Revenue (€m)"]) or pd.isna(row["EBITDA (%)"]):
        st.warning("Scenario needs Revenue (€m) and EBITDA (%) in the snapshot dataset.")
    else:
        rev = float(row["Revenue (€m)"])
        ebitda_pct = float(row["EBITDA (%)"])
        baseline_ebita = rev * (ebitda_pct/100.0)

        # revenue scales with rate + utilization
        rev_s = rev * (1 + rate_delta/100.0) * (1 + util_delta/100.0)

        # cost proxy: assume baseline cost = rev - ebita; wage inflation applies to 60% of cost base (generic public assumption)
        baseline_cost = rev - baseline_ebita
        wage_cost_share = 0.60
        scenario_cost = (baseline_cost * (1 - wage_cost_share) * (rev_s/rev)) + (baseline_cost * wage_cost_share * (rev_s/rev) * (1 + wage_infl/100.0))

        scenario_ebita = rev_s - scenario_cost
        delta = scenario_ebita - baseline_ebita

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline revenue", fmt_meur(rev))
            st.metric("Scenario revenue", fmt_meur(rev_s))
        with c2:
            st.metric("Baseline EBITA proxy", fmt_meur(baseline_ebita))
            st.metric("Scenario EBITA proxy", fmt_meur(scenario_ebita))
        with c3:
            st.metric("EBITA impact", fmt_meur(delta))

with tabs[2]:
    st.markdown("### Data & export")
    st.download_button("benchmark_snapshot_public.csv", data=df_snap_all.to_csv(index=False), file_name="benchmark_snapshot_public.csv", mime="text/csv")
    st.download_button("benchmark_trends_public.csv", data=df_trends_all.to_csv(index=False), file_name="benchmark_trends_public.csv", mime="text/csv")
    st.code("See schema.md for the public-only data model.", language="text")
