import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Market Benchmark – Siili Solutions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET = "Siili Solutions"

# -------------------------------------------------
# UI labels (no underscores / no pct)
# -------------------------------------------------
LABELS: Dict[str, str] = {
    "year": "Year",
    "company": "Company",
    "country_group": "Peer group",
    "peer_bucket": "Group",
    "data_confidence": "Confidence",
    "revenue_meur": "Revenue (€m)",
    "ebitda_pct": "EBITDA (%)",
    "personnel_cost_pct": "Personnel cost (% of revenue)",
    "outsourcing_pct": "Outsourcing (% of revenue)",
    "billable_pct": "Billable (%)",
    "senior_pct": "Senior (%)",
    "offshore_pct": "Offshore (%)",
    "headcount": "Headcount",
    "revenue_per_employee_keur": "Revenue / Employee (k€)",
    "service_focus": "Service focus",
    "delivery_model": "Delivery model",
    "industry_focus": "Industry focus",
    "scale_bucket": "Scale bucket",
}

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

# -------------------------------------------------
# Metrics
# -------------------------------------------------
@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    good_if_higher: bool
    fmt: str  # pct|keur|meur

METRICS: List[MetricDef] = [
    MetricDef("ebitda_pct", "EBITDA (%)", True, "pct"),
    MetricDef("personnel_cost_pct", "Personnel cost (% of revenue)", False, "pct"),
    MetricDef("revenue_per_employee_keur", "Revenue / Employee (k€)", True, "keur"),
    MetricDef("outsourcing_pct", "Outsourcing (% of revenue)", False, "pct"),
]

TOOLTIPS = {
    "ebitda_pct": "EBITDA margin (%) – higher is better.",
    "personnel_cost_pct": "Personnel cost as % of revenue – lower is better.",
    "revenue_per_employee_keur": "Revenue divided by headcount – higher is better.",
    "outsourcing_pct": "Outsourcing/subcontracting as % of revenue – lower is shown as better for comparability.",
}

# -------------------------------------------------
# Data loading
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snap = load_csv("data/benchmark_snapshot.csv")
    trends = load_csv("data/benchmark_trends.csv")
    analyst = load_csv("data/analyst_estimates.csv")
    offering = load_csv("data/strategic_offering.csv")
    peer_offer = load_csv("data/peer_offering_heatmap.csv")
    comp_int = load_csv("data/competitive_intensity.csv")
    return snap, trends, analyst, offering, peer_offer, comp_int

df_snap_all, df_trends_all, df_analyst, df_offering, df_peer_offer, df_comp_int = load_all()

# numeric coercion
NUM_COLS = [
    "year","revenue_meur","ebitda_pct","personnel_cost_pct","outsourcing_pct","headcount",
    "billable_pct","senior_pct","offshore_pct","revenue_per_employee_keur"
]
for c in NUM_COLS:
    if c in df_snap_all.columns:
        df_snap_all[c] = pd.to_numeric(df_snap_all[c], errors="coerce")
    if c in df_trends_all.columns:
        df_trends_all[c] = pd.to_numeric(df_trends_all[c], errors="coerce")
for c in ["pricing_pressure_1to5","regulated_industries_1to3","delivery_scale_1to3","competitive_intensity_index"]:
    if c in df_comp_int.columns:
        df_comp_int[c] = pd.to_numeric(df_comp_int[c], errors="coerce")

if "revenue_per_employee_keur" not in df_trends_all.columns and "headcount" in df_trends_all.columns:
    df_trends_all["revenue_per_employee_keur"] = (df_trends_all["revenue_meur"]*1000) / df_trends_all["headcount"]

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.markdown("## Controls")

presentation_mode = st.sidebar.toggle("Presentation mode (minimal UI)", value=False, help="Hides sidebar & technical labels for executive presenting.")
if presentation_mode:
    st.markdown("""
    <style>
      [data-testid="stSidebar"] {display:none;}
      [data-testid="stSidebarNav"] {display:none;}
      .block-container {padding-top: 1.2rem;}
    </style>
    """, unsafe_allow_html=True)

_years = pd.to_numeric(df_snap_all["year"], errors="coerce").dropna().astype(int).unique().tolist()
_years = sorted(_years)
if len(_years) == 0:
    st.sidebar.error("No valid years found in data/benchmark_snapshot.csv (column: year).")
    st.stop()
elif len(_years) == 1:
    snapshot_year = int(_years[0])
    st.sidebar.info(f"Snapshot year fixed to {snapshot_year} (only one year in dataset).")
else:
    snapshot_year = int(st.sidebar.select_slider("Snapshot year", options=_years, value=_years[-1]))

peer_groups = st.sidebar.multiselect(
    "Peer groups",
    options=["Finland", "International"],
    default=["Finland", "International"],
    help="Choose which peer groups to include in medians and visuals.",
)

include_estimated = st.sidebar.toggle(
    "Include estimated / inferred data",
    value=True,
    help="If off, keeps only rows where Confidence = public.",
)

company_drill = st.sidebar.selectbox(
    "Drill-down company",
    options=sorted(df_snap_all["company"].dropna().unique()),
    index=sorted(df_snap_all["company"].dropna().unique()).index(TARGET) if TARGET in df_snap_all["company"].unique() else 0
)

trend_metric_key = st.sidebar.selectbox(
    "Trend metric",
    options=[m.key for m in METRICS] + ["revenue_meur", "headcount", "billable_pct", "senior_pct", "offshore_pct"],
    index=0,
)

show_forecasts = st.sidebar.toggle("Show forecasts (next 2 years)", value=True, help="Shows inferred forecasts in the trend chart.")
scatter_labels = st.sidebar.toggle("2x2: show labels", value=False)
scatter_zoom = st.sidebar.slider("2x2: zoom", 70, 220, 130)

# Scenario sliders
st.sidebar.markdown("---")
st.sidebar.markdown("## Scenario sliders (EBITA impact)")
util_delta = st.sidebar.slider("Utilization change (pp)", -10.0, 10.0, 0.0, 0.5)
rate_delta = st.sidebar.slider("Billing rate change (%)", -10.0, 10.0, 0.0, 0.5)
wage_infl = st.sidebar.slider("Wage inflation (%)", 0.0, 10.0, 3.0, 0.5)

# -------------------------------------------------
# Filters
# -------------------------------------------------
df_snap = df_snap_all[df_snap_all["year"] == snapshot_year].copy()

if not include_estimated and "data_confidence" in df_snap.columns:
    df_snap = df_snap[df_snap["data_confidence"].fillna("estimated") == "public"].copy()

df_snap["peer_bucket"] = np.where(df_snap["company"] == TARGET, "Siili", df_snap["country_group"])
selected_buckets = ["Siili"] + peer_groups
df_snap_f = df_snap[df_snap["peer_bucket"].isin(selected_buckets)].copy()

df_trends = df_trends_all.copy()
if not include_estimated and "data_confidence" in df_trends.columns:
    df_trends = df_trends[df_trends["data_confidence"].fillna("estimated") == "public"].copy()
if not show_forecasts:
    df_trends = df_trends[df_trends["year"] <= snapshot_year].copy()
df_trends_f = df_trends[df_trends["country_group"].isin(peer_groups + ["Target"])].copy()

# -------------------------------------------------
# Medians
# -------------------------------------------------
def median_for(group: str, key: str) -> float:
    if group == "Siili":
        s = df_snap_f[df_snap_f["company"] == TARGET][key]
    else:
        s = df_snap_f[df_snap_f["country_group"] == group][key]
    return float(s.median()) if len(s.dropna()) else float("nan")

# -------------------------------------------------
# Archetypes (simple, explainable rules)
# -------------------------------------------------
def assign_archetype(r: pd.Series) -> str:
    if r.get("company") == TARGET:
        return "Target"
    out = float(r.get("outsourcing_pct", np.nan))
    off = float(r.get("offshore_pct", np.nan))
    rev = float(r.get("revenue_meur", np.nan))
    cg = r.get("country_group", "")
    # rules
    if cg == "International" and (off >= 15 or out >= 30):
        return "Scale / nearshore"
    if out >= 35:
        return "Managed services / flexible capacity"
    if rev >= 500:
        return "Incumbent scale"
    return "Local premium builder"

df_snap_f["archetype"] = df_snap_f.apply(assign_archetype, axis=1)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
      <div style="font-size:46px;font-weight:800;letter-spacing:-0.5px;">Market Benchmark</div>
      <div style="font-size:22px;font-weight:700;padding:10px 16px;border-radius:999px;background:rgba(27,163,255,0.18);border:1px solid rgba(27,163,255,0.35);">
        Siili Solutions
      </div>
    </div>
    <div style="opacity:0.82;margin-top:-8px;">
      CEO-ready • green = good • red = bad • trends + forecasts • drill-down • snapshot year: <b>{snapshot_year}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# KPI cards
# -------------------------------------------------
siili_row = df_snap_f[df_snap_f["company"] == TARGET]
siili = siili_row.iloc[0] if len(siili_row) else None

cols = st.columns(4)
for i, m in enumerate(METRICS):
    with cols[i]:
        si_val = float(siili[m.key]) if siili is not None and pd.notna(siili[m.key]) else float("nan")
        ref = median_for("Finland", m.key) if "Finland" in peer_groups else float("nan")
        delta = si_val - ref if not np.isnan(ref) else float("nan")

        if m.fmt == "pct":
            main = fmt_pct(si_val, 1)
            delta_txt = f"{delta:+.1f} pp" if not np.isnan(delta) else "–"
        elif m.fmt == "keur":
            main = fmt_keur(si_val, 0)
            delta_txt = f"{delta:+.0f}k€" if not np.isnan(delta) else "–"
        else:
            main = fmt_meur(si_val, 1)
            delta_txt = f"{delta:+.1f}€m" if not np.isnan(delta) else "–"

        c = delta_color(delta, m.good_if_higher)
        st.markdown(
            f"""
            <div style="padding:18px 18px 14px 18px;border-radius:18px;background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.08);box-shadow:0 8px 30px rgba(0,0,0,0.35);">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-size:14px;opacity:0.88;">{m.label}</div>
                <div title="{TOOLTIPS.get(m.key,'')}" style="font-size:14px;opacity:0.6;">ⓘ</div>
              </div>
              <div style="font-size:44px;font-weight:800;line-height:1;margin-top:10px;">{main}</div>
              <div style="font-size:18px;font-weight:800;margin-top:8px;color:{c};">{delta_txt}</div>
              <div style="opacity:0.66;font-size:12px;margin-top:10px;">Delta vs Finland median (direction-aware)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------
# Executive storytelling: "So what?" + "Action"
# -------------------------------------------------
def so_what_and_action() -> Tuple[str, str]:
    if siili is None or "Finland" not in peer_groups:
        return ("Use the peer filters to compare against Finland or International medians.",
                "Confirm which peer group you want to anchor the narrative to, then prioritize 1–2 improvement levers.")
    # Compute a few gaps
    e_gap = float(siili["ebitda_pct"]) - median_for("Finland", "ebitda_pct")
    rpe_gap = float(siili["revenue_per_employee_keur"]) - median_for("Finland", "revenue_per_employee_keur")
    out_gap = float(siili["outsourcing_pct"]) - median_for("Finland", "outsourcing_pct")

    # Rule-based narrative
    if e_gap < -1.0 and rpe_gap < -5:
        so = "Siili underperforms the Finland peer median in both EBITDA and productivity (Revenue/Employee). This typically points to utilization/rate and delivery mix as the key drivers."
        act = "Action: focus the next 90 days on (1) improving utilization, (2) selective rate management on strong accounts, and (3) tightening subcontractor governance to protect margin."
    elif e_gap >= 0 and rpe_gap >= 0:
        so = "Siili sits at or above the Finland peer median in both EBITDA and productivity—this is a strong positioning baseline."
        act = "Action: protect the moat by doubling down on the strongest service lines, and invest in repeatable offerings to scale without margin dilution."
    elif out_gap > 8 and e_gap < 0:
        so = "Outsourcing share is notably above the Finland peer median while EBITDA lags—this combination can indicate margin leakage through subcontracting or suboptimal mix."
        act = "Action: improve subcontractor pricing discipline, increase internal billable mix, and standardize delivery governance for outsourced work."
    else:
        so = "Siili’s gaps vs peers are mixed: prioritize the one lever that improves both margin and productivity."
        act = "Action: run a short driver review (utilization, rate, senior mix, subcontracting) and pick 1–2 levers with measurable monthly KPIs."
    return so, act

so, act = so_what_and_action()
st.markdown("### Executive summary")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"**So what?**  \n{so}")
with c2:
    st.markdown(f"**Recommended action**  \n{act}")

# -------------------------------------------------
# Trends
# -------------------------------------------------
st.markdown("### Trends")
df_t = df_trends_f.copy()
if trend_metric_key == "revenue_per_employee_keur":
    df_t["revenue_per_employee_keur"] = (df_t["revenue_meur"]*1000) / df_t["headcount"]

si = df_t[df_t["company"] == TARGET].groupby("year")[trend_metric_key].median().reset_index(name="value")
fi = df_t[df_t["country_group"] == "Finland"].groupby("year")[trend_metric_key].median().reset_index(name="value")
intl = df_t[df_t["country_group"] == "International"].groupby("year")[trend_metric_key].median().reset_index(name="value")

fig = go.Figure()
if len(fi):
    fig.add_trace(go.Scatter(x=fi["year"], y=fi["value"], mode="lines+markers", name="Finland median", line=dict(dash="dot")))
if len(intl):
    fig.add_trace(go.Scatter(x=intl["year"], y=intl["value"], mode="lines+markers", name="International median", line=dict(dash="dot")))
if len(si):
    fig.add_trace(go.Scatter(x=si["year"], y=si["value"], mode="lines+markers", name="Siili Solutions", line=dict(width=4)))

if show_forecasts:
    max_hist = snapshot_year
    max_all = int(df_t["year"].max())
    if max_all > max_hist:
        fig.add_vrect(x0=max_hist+0.5, x1=max_all+0.5, fillcolor="rgba(255,255,255,0.06)", line_width=0)

fig.update_layout(
    height=330,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=-0.25),
    xaxis_title="Year",
    yaxis_title=LABELS.get(trend_metric_key, trend_metric_key),
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Tabs: keep v6 features + improved
# -------------------------------------------------
tabs = st.tabs([
    "Cost structure",
    "Drivers radar",
    "Strategic positioning (2x2)",
    "Peer-offering heatmap",
    "Competitive intensity",
    "Scenario (EBITA impact)",
    "Analyst expectations",
    "Data & export",
])

def insights_box(title: str, bullets: List[str]):
    st.markdown(f"#### {title}")
    for b in bullets:
        st.markdown(f"- {b}")

# Cost structure
with tabs[0]:
    st.markdown("### Cost structure (% of revenue)")
    d = df_snap_f.sort_values(["country_group","company"]).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["company"], y=d["personnel_cost_pct"], name="Personnel cost"))
    fig.add_trace(go.Bar(x=d["company"], y=d["outsourcing_pct"], name="Outsourcing"))
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        height=430,
        margin=dict(l=10,r=10,t=10,b=10),
        yaxis_title="% of revenue",
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)
    if siili is not None and "Finland" in peer_groups:
        insights_box("Insights", [
            f"Personnel cost gap vs Finland median: **{float(siili['personnel_cost_pct'])-median_for('Finland','personnel_cost_pct'):+.1f} pp** (lower is better).",
            f"Outsourcing gap vs Finland median: **{float(siili['outsourcing_pct'])-median_for('Finland','outsourcing_pct'):+.1f} pp** (lower is better).",
            "Use together with EBITDA: higher outsourcing with lower EBITDA may indicate margin leakage or pricing/mix issues.",
        ])

# Drivers radar
with tabs[1]:
    st.markdown("### Profitability drivers radar")
    dims = [("billable_pct","Billable"), ("senior_pct","Senior"), ("outsourcing_pct","Outsourcing"), ("offshore_pct","Offshore")]
    fig = go.Figure()
    for _, r in df_snap_f.iterrows():
        vals = [float(r[k]) if pd.notna(r[k]) else 0.0 for k,_ in dims]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[lab for _,lab in dims] + [dims[0][1]],
            fill="toself" if r["company"]==TARGET else "none",
            name=r["company"],
            opacity=0.9 if r["company"]==TARGET else 0.55,
        ))
    fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10,r=10,t=10,b=10),
                      polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

# 2x2 improved + archetypes
with tabs[2]:
    st.markdown("### Strategic positioning (2x2)")
    d = df_snap_f.copy()
    d["label"] = np.where(d["company"]==TARGET, d["company"], np.where(scatter_labels, d["company"], ""))

    x = d["revenue_per_employee_keur"]
    y = d["ebitda_pct"]
    x_med = float(np.nanmedian(x))
    y_med = float(np.nanmedian(y))
    x_span = (np.nanmax(x)-np.nanmin(x))*(scatter_zoom/100.0) if np.nanmax(x)>np.nanmin(x) else 10
    y_span = (np.nanmax(y)-np.nanmin(y))*(scatter_zoom/100.0) if np.nanmax(y)>np.nanmin(y) else 5
    xr = [max(0, x_med - x_span/2), x_med + x_span/2]
    yr = [y_med - y_span/2, y_med + y_span/2]

    fig = px.scatter(
        d,
        x="revenue_per_employee_keur",
        y="ebitda_pct",
        size="revenue_meur",
        color="archetype",
        symbol="peer_bucket",
        text="label",
        hover_name="company",
        hover_data={
            "data_confidence": True,
            "country_group": True,
            "revenue_meur":":.1f",
            "headcount":True,
            "personnel_cost_pct":":.1f",
            "outsourcing_pct":":.1f",
            "billable_pct":":.1f",
            "senior_pct":":.1f",
            "offshore_pct":":.1f",
        },
        labels={
            "revenue_per_employee_keur": "Revenue / Employee (k€)",
            "ebitda_pct": "EBITDA (%)",
            "archetype": "Archetype",
            "peer_bucket": "Group",
        }
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.35)")), textposition="top center")
    if "Finland" in peer_groups:
        fig.add_vline(x=median_for("Finland","revenue_per_employee_keur"), line_dash="dot", line_color="rgba(255,255,255,0.35)")
        fig.add_hline(y=median_for("Finland","ebitda_pct"), line_dash="dot", line_color="rgba(255,255,255,0.35)")
    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=10,r=10,t=10,b=10),
                      legend=dict(orientation="h", y=-0.2),
                      xaxis=dict(range=xr), yaxis=dict(range=yr))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Color = competitor archetype (explainable rules). Symbol = Siili vs peer group. Hover shows confidence tag.")

# Heatmap
with tabs[3]:
    st.markdown("### Peer-offering heatmap (capabilities × companies)")
    vis_companies = sorted(df_snap_f["company"].unique().tolist())
    heat = df_peer_offer[df_peer_offer["company"].isin(vis_companies)].copy()
    piv = heat.pivot_table(index="company", columns="capability", values="focus_score_0to3", aggfunc="max").reindex(vis_companies).fillna(0)
    fig = px.imshow(piv, aspect="auto", labels=dict(x="Capability", y="Company", color="Focus (0–3)"), height=560)
    fig.update_layout(template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Scores are heuristic; update peer_offering_heatmap.csv to reflect validated research.")

# Competitive intensity
with tabs[4]:
    st.markdown("### Competitive intensity")
    ci = df_comp_int.merge(df_offering[["company","delivery_model","industry_focus","scale_bucket"]], on="company", how="left")
    ci = ci[ci["company"].isin(df_snap_f["company"])].copy()
    fig = px.scatter(
        ci,
        x="pricing_pressure_1to5",
        y="competitive_intensity_index",
        color="country_group",
        size="delivery_scale_1to3",
        hover_name="company",
        hover_data={"regulated_industries_1to3":True,"delivery_model":True,"industry_focus":True,"scale_bucket":True,"confidence":True},
        labels={
            "pricing_pressure_1to5": "Pricing pressure (1–5)",
            "competitive_intensity_index": "Competitive intensity",
            "country_group": "Peer group",
            "delivery_scale_1to3": "Delivery scale",
        },
        height=460,
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ci.sort_values("competitive_intensity_index", ascending=False), use_container_width=True, hide_index=True)

# Scenario model
def scenario_ebita_impact(si: pd.Series) -> dict:
    if si is None:
        return {"baseline": np.nan, "scenario": np.nan, "delta": np.nan}
    rev = float(si.get("revenue_meur", np.nan))
    ebitda_pct = float(si.get("ebitda_pct", np.nan))
    personnel_pct = float(si.get("personnel_cost_pct", np.nan))
    if any(np.isnan(x) for x in [rev, ebitda_pct, personnel_pct]):
        return {"baseline": np.nan, "scenario": np.nan, "delta": np.nan}

    baseline = rev * (ebitda_pct/100.0)

    rev_s = rev * (1 + rate_delta/100.0) * (1 + util_delta/100.0)

    baseline_total_cost = rev - baseline
    baseline_personnel_cost = (personnel_pct/100.0) * rev
    other_cost = max(0.0, baseline_total_cost - baseline_personnel_cost)

    scenario_personnel_cost = baseline_personnel_cost * (1 + wage_infl/100.0) * (rev_s/rev)
    scenario_total_cost = other_cost * (rev_s/rev) + scenario_personnel_cost

    scenario = rev_s - scenario_total_cost
    return {"baseline": baseline, "scenario": scenario, "delta": scenario-baseline, "rev0": rev, "rev1": rev_s}

with tabs[5]:
    st.markdown("### Scenario (EBITA impact – directional)")
    impact = scenario_ebita_impact(siili)
    if np.isnan(impact["baseline"]):
        st.warning("Scenario model needs Revenue, EBITDA (%) and Personnel cost (% of revenue) for Siili in snapshot data.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline Revenue", fmt_meur(impact["rev0"]))
            st.metric("Scenario Revenue", fmt_meur(impact["rev1"]))
        with c2:
            st.metric("Baseline EBITA (proxy)", fmt_meur(impact["baseline"]))
            st.metric("Scenario EBITA (proxy)", fmt_meur(impact["scenario"]))
        with c3:
            st.metric("EBITA impact", fmt_meur(impact["delta"]))
        insights_box("Assumptions", [
            "Revenue scales with utilization change and billing rate change.",
            "Personnel cost scales with revenue and wage inflation; other costs scale with revenue.",
            "This is a **directional** executive model. Replace with a full driver-based P&L when internal data is available."
        ])

# Analyst expectations (kept)
with tabs[6]:
    st.markdown("### Analyst expectations (Siili)")
    st.caption("Update data/analyst_estimates.csv as new reports are published.")
    df_a = df_analyst.copy()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.line(df_a, x="year", y="revenue_meur", color="source", markers=True, labels={"year":"Year","revenue_meur":"Revenue (€m)"})
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(df_a, x="year", y="ebita_pct", color="source", markers=True, labels={"year":"Year","ebita_pct":"EBITA (%)"})
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_a.sort_values(["source","year"]), use_container_width=True, hide_index=True)

# Data & export (improved)
with tabs[7]:
    st.markdown("### Data & export")
    st.write("Download datasets used by the dashboard:")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("benchmark_snapshot.csv", data=df_snap_all.to_csv(index=False), file_name="benchmark_snapshot.csv", mime="text/csv")
    with c2:
        st.download_button("benchmark_trends.csv", data=df_trends_all.to_csv(index=False), file_name="benchmark_trends.csv", mime="text/csv")
    with c3:
        st.download_button("analyst_estimates.csv", data=df_analyst.to_csv(index=False), file_name="analyst_estimates.csv", mime="text/csv")
    st.download_button("peer_offering_heatmap.csv", data=df_peer_offer.to_csv(index=False), file_name="peer_offering_heatmap.csv", mime="text/csv")
    st.download_button("competitive_intensity.csv", data=df_comp_int.to_csv(index=False), file_name="competitive_intensity.csv", mime="text/csv")

    st.markdown("#### Quick share pack (HTML)")
    st.caption("Exports a lightweight HTML snapshot (charts as interactive Plotly JSON) that you can attach/share.")
    # Minimal HTML bundle: summary numbers + note (avoid kaleido dependency)
    summary = {
        "snapshot_year": snapshot_year,
        "kpis": {m.label: (float(siili[m.key]) if siili is not None and pd.notna(siili[m.key]) else None) for m in METRICS},
        "peer_groups": peer_groups,
        "include_estimated": include_estimated,
        "notes": "This is an auto-generated share snapshot. For full interactivity, use the live Streamlit link.",
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Market Benchmark – Share Pack</title></head>
<body style="font-family:system-ui;max-width:900px;margin:24px auto;">
<h1>Market Benchmark – Siili Solutions</h1>
<p><b>Snapshot year:</b> {snapshot_year}</p>
<h2>Executive KPI summary</h2>
<pre style="background:#f6f8fa;padding:16px;border-radius:10px;">{pd.Series(summary['kpis']).to_string()}</pre>
<p><b>Peer groups:</b> {", ".join(peer_groups) if peer_groups else "–"}</p>
<p><b>Include estimated:</b> {include_estimated}</p>
<p style="opacity:0.7">{summary['notes']}</p>
</body></html>"""
    st.download_button("Download share_pack.html", data=html.encode("utf-8"), file_name="share_pack.html", mime="text/html")
