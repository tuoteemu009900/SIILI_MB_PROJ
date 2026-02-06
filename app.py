
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Market Benchmark – Siili Solutions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling helpers
# ----------------------------
def _metric_delta_color(value: float, good_if_higher: bool = True) -> str:
    """
    Returns a hex color:
    - green for "good" direction
    - red for "bad" direction
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "#9AA4B2"
    good = value >= 0 if good_if_higher else value <= 0
    return "#2ED573" if good else "#FF4757"

def format_pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{decimals}f}%"

def format_eur_meur(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{decimals}f}€m"

def format_keur(x: float, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{decimals}f}k€"

def safe_div(a: float, b: float) -> float:
    if b in (0, None) or (isinstance(b, float) and np.isnan(b)):
        return float("nan")
    return a / b

@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    unit: str
    good_if_higher: bool
    fmt: str  # "pct" | "keur" | "meur"

METRICS: List[MetricDef] = [
    MetricDef("ebitda_pct", "EBITDA", "%", True, "pct"),
    MetricDef("personnel_cost_pct", "Personnel cost", "%", False, "pct"),
    MetricDef("revenue_per_employee_keur", "Revenue / Employee", "k€", True, "keur"),
    MetricDef("outsourcing_pct", "Outsourcing", "%", False, "pct"),
]

TOOLTIP_HELP = {
    "ebitda_pct": "EBITDA margin (%) – higher is better.",
    "personnel_cost_pct": "Personnel cost as % of revenue – lower is better.",
    "revenue_per_employee_keur": "Revenue / headcount (k€).",
    "outsourcing_pct": "Outsourcing/subcontracting as % of revenue – depends on model; in this dashboard lower is shown as better.",
}

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snap = load_csv("data/benchmark_snapshot.csv")
    trends = load_csv("data/benchmark_trends.csv")
    analyst = load_csv("data/analyst_estimates.csv")
    offering = load_csv("data/strategic_offering.csv")
    return snap, trends, analyst, offering

df_snap_all, df_trends_all, df_analyst, df_offering = load_all()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.markdown("## Controls")
snapshot_year = int(st.sidebar.slider("Snapshot year", min_value=int(df_snap_all["year"].min()), max_value=int(df_snap_all["year"].max()), value=int(df_snap_all["year"].max()), step=1))

peer_groups = st.sidebar.multiselect(
    "Peer groups",
    options=["Finland", "International"],
    default=["Finland", "International"],
    help="Choose which peer groups to include in medians and visuals.",
)

include_estimated = st.sidebar.toggle("Include estimated/imputed", value=True, help="If off, hides rows marked as estimated/imputed.")
company_drill = st.sidebar.selectbox("Drill-down company", options=sorted(df_snap_all["company"].unique()), index=sorted(df_snap_all["company"].unique()).index("Siili Solutions"))

trend_metric_key = st.sidebar.selectbox(
    "Trend metric",
    options=[m.key for m in METRICS] + ["revenue_meur", "headcount", "billable_pct", "senior_pct", "offshore_pct"],
    index=0,
)

# ----------------------------
# Filtered data
# ----------------------------
df_snap = df_snap_all[df_snap_all["year"] == snapshot_year].copy()
df_trends = df_trends_all.copy()
df_trends = df_trends[df_trends["year"].between(df_trends_all["year"].min(), snapshot_year)]

if not include_estimated:
    df_snap = df_snap[df_snap["estimated_imputed"] == 0]
    df_trends = df_trends[df_trends["estimated_imputed"] == 0]

df_snap["peer_bucket"] = np.where(df_snap["company"] == "Siili Solutions", "Siili", df_snap["country_group"])

selected_buckets = ["Siili"] + peer_groups
df_snap_f = df_snap[df_snap["peer_bucket"].isin(selected_buckets)].copy()
df_trends_f = df_trends[df_trends["country_group"].isin(peer_groups + ["Target"])].copy()

# Median references
def median_for(bucket: str, key: str) -> float:
    if bucket == "Siili":
        s = df_snap_f[df_snap_f["company"] == "Siili Solutions"][key]
    elif bucket == "Finland":
        s = df_snap_f[df_snap_f["country_group"] == "Finland"][key]
    else:
        s = df_snap_f[df_snap_f["country_group"] == "International"][key]
    return float(s.median()) if len(s) else float("nan")

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
      <div style="font-size:46px;font-weight:800;letter-spacing:-0.5px;">Market Benchmark</div>
      <div style="font-size:22px;font-weight:700;padding:10px 16px;border-radius:999px;background:rgba(27,163,255,0.18);border:1px solid rgba(27,163,255,0.35);">
        Siili Solutions
      </div>
    </div>
    <div style="opacity:0.8;margin-top:-8px;">CEO-ready • green = good • red = bad • trends + drill-down • snapshot year: <b>"""
    + str(snapshot_year)
    + """</b></div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Executive KPI summary (cards)
# ----------------------------
siili = df_snap_f[df_snap_f["company"] == "Siili Solutions"].iloc[0] if (df_snap_f["company"] == "Siili Solutions").any() else None
fi_median = df_snap_f[df_snap_f["country_group"] == "Finland"]
int_median = df_snap_f[df_snap_f["country_group"] == "International"]

card_cols = st.columns(4)
for i, m in enumerate(METRICS):
    with card_cols[i]:
        si_val = float(siili[m.key]) if siili is not None else float("nan")
        ref = median_for("Finland", m.key) if "Finland" in peer_groups else float("nan")
        delta = si_val - ref if not np.isnan(ref) else float("nan")

        if m.fmt == "pct":
            main = format_pct(si_val, 1)
            delta_txt = f"{delta:+.1f} pp" if not np.isnan(delta) else "–"
        elif m.fmt == "keur":
            main = format_keur(si_val, 0)
            delta_txt = f"{delta:+.0f}k€" if not np.isnan(delta) else "–"
        else:
            main = format_eur_meur(si_val, 1)
            delta_txt = f"{delta:+.1f}€m" if not np.isnan(delta) else "–"

        delta_color = _metric_delta_color(delta, good_if_higher=m.good_if_higher)

        st.markdown(
            f"""
            <div style="padding:18px 18px 14px 18px;border-radius:18px;background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.08);box-shadow:0 8px 30px rgba(0,0,0,0.35);">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-size:14px;opacity:0.85;">{m.label}</div>
                <div title="{TOOLTIP_HELP.get(m.key,'')}" style="font-size:14px;opacity:0.6;">ⓘ</div>
              </div>
              <div style="font-size:44px;font-weight:800;line-height:1;margin-top:10px;">{main}</div>
              <div style="font-size:18px;font-weight:700;margin-top:8px;color:{delta_color};">{delta_txt}</div>
              <div style="opacity:0.65;font-size:12px;margin-top:10px;">Delta vs Finland median (direction-aware)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------
# Executive insights + trend
# ----------------------------
left, right = st.columns([1.25, 1.0], gap="large")

def build_exec_insights() -> List[str]:
    bullets = []
    if siili is None:
        return ["No Siili row found in snapshot data."]
    for m in METRICS:
        si_val = float(siili[m.key])
        if "Finland" in peer_groups:
            ref_fi = median_for("Finland", m.key)
            d = si_val - ref_fi
            sign = "+" if d >= 0 else "–"
            if m.fmt == "pct":
                bullets.append(f"**{m.label}:** Siili {si_val:.1f}% vs FI median {ref_fi:.1f}% → {d:+.1f} pp.")
            elif m.fmt == "keur":
                bullets.append(f"**{m.label}:** Siili {si_val:.0f}k€ vs FI median {ref_fi:.0f}k€ → {d:+.0f}k€.")
            else:
                bullets.append(f"**{m.label}:** Siili {si_val:.1f}€m vs FI median {ref_fi:.1f}€m → {d:+.1f}€m.")
        if "International" in peer_groups:
            ref_int = median_for("International", m.key)
            d2 = si_val - ref_int
            if m.fmt == "pct":
                bullets.append(f"**{m.label}:** Siili {si_val:.1f}% vs INT median {ref_int:.1f}% → {d2:+.1f} pp.")
            elif m.fmt == "keur":
                bullets.append(f"**{m.label}:** Siili {si_val:.0f}k€ vs INT median {ref_int:.0f}k€ → {d2:+.0f}k€.")
            else:
                bullets.append(f"**{m.label}:** Siili {si_val:.1f}€m vs INT median {ref_int:.1f}€m → {d2:+.1f}€m.")
    # data quality
    share_est = float(df_snap_f["estimated_imputed"].mean()) if len(df_snap_f) else 0.0
    bullets.append(f"**Data quality:** {share_est*100:.0f}% rows are *estimated/imputed* in this demo dataset.")
    return bullets

with left:
    st.markdown("### Executive insights")
    for b in build_exec_insights():
        st.markdown(f"- {b}")

with right:
    st.markdown("### Trends")
    # prepare trend series: Siili + medians
    df_t = df_trends_f.copy()
    # add derived metric if needed
    if trend_metric_key == "revenue_per_employee_keur":
        df_t["revenue_per_employee_keur"] = (df_t["revenue_meur"]*1000) / df_t["headcount"]

    # series
    si = df_t[df_t["company"] == "Siili Solutions"].groupby("year")[trend_metric_key].median().reset_index(name="value")
    fi = df_t[df_t["country_group"] == "Finland"].groupby("year")[trend_metric_key].median().reset_index(name="value")
    intl = df_t[df_t["country_group"] == "International"].groupby("year")[trend_metric_key].median().reset_index(name="value")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fi["year"], y=fi["value"], mode="lines+markers", name="Finland (Peer) median", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=intl["year"], y=intl["value"], mode="lines+markers", name="International (Peer) median", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=si["year"], y=si["value"], mode="lines+markers", name="Siili Solutions", line=dict(width=4)))

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.25),
        xaxis_title="Year",
        yaxis_title=trend_metric_key,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main sections in tabs
# ----------------------------
tabs = st.tabs(["Cost Structure", "Drivers Radar", "Strategic Positioning (2x2)", "Analyst Expectations", "Strategy & Market Map", "Data & Export"])

def _direction_aware_delta(metric_key: str) -> bool:
    # good_if_higher for KPI; used only for delta styling
    md = next((m for m in METRICS if m.key == metric_key), None)
    return md.good_if_higher if md else True

def build_cost_structure(df: pd.DataFrame) -> go.Figure:
    # stacked bar % revenue
    d = df.copy()
    d = d.sort_values(["country_group","company"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["company"],
        y=d["personnel_cost_pct"],
        name="Personnel cost (%)",
        hovertemplate="<b>%{x}</b><br>Personnel: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=d["company"],
        y=d["outsourcing_pct"],
        name="Outsourcing (%)",
        hovertemplate="<b>%{x}</b><br>Outsourcing: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        height=430,
        margin=dict(l=10,r=10,t=10,b=10),
        yaxis_title="% of revenue",
        xaxis_title="",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig

def build_radar(df: pd.DataFrame) -> go.Figure:
    dims = [("billable_pct","Billable"), ("senior_pct","Senior"), ("outsourcing_pct","Outsourcing"), ("offshore_pct","Offshore")]
    fig = go.Figure()
    for _,r in df.iterrows():
        vals = [float(r[k]) for k,_ in dims]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[lab for _,lab in dims] + [dims[0][1]],
            fill="toself" if r["company"]=="Siili Solutions" else "none",
            name=r["company"],
            opacity=0.9 if r["company"]=="Siili Solutions" else 0.6,
            hovertemplate="<b>%{text}</b><br>" + "<br>".join([f"{lab}: %{{r[{i}]:.1f}}%" for i,(_,lab) in enumerate(dims)]) + "<extra></extra>",
            text=[r["company"]]*(len(dims)+1),
        ))
    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=10,r=10,t=10,b=10),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig

def build_2x2(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    fig = px.scatter(
        d,
        x="revenue_per_employee_keur",
        y="ebitda_pct",
        size="revenue_meur",
        color="peer_bucket",
        hover_name="company",
        hover_data={
            "revenue_meur":":.1f",
            "headcount":True,
            "personnel_cost_pct":":.1f",
            "outsourcing_pct":":.1f",
            "billable_pct":":.1f",
            "senior_pct":":.1f",
            "offshore_pct":":.1f",
            "revenue_per_employee_keur":":.0f",
            "ebitda_pct":":.1f",
            "peer_bucket":False,
        },
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.35)")))
    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="Revenue per Employee (k€)",
        yaxis_title="EBITDA (%)",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig

def section_ai_insight(title: str, bullets: List[str]):
    st.markdown(f"#### {title} – AI insights (rule-based)")
    for b in bullets:
        st.markdown(f"- {b}")

with tabs[0]:
    st.markdown("### Cost structure (% of revenue)")
    st.plotly_chart(build_cost_structure(df_snap_f), use_container_width=True)

    # Insight
    si_p = float(siili["personnel_cost_pct"]) if siili is not None else float("nan")
    si_o = float(siili["outsourcing_pct"]) if siili is not None else float("nan")
    fi_p = median_for("Finland","personnel_cost_pct")
    fi_o = median_for("Finland","outsourcing_pct")
    bullets = [
        f"Personnel cost delta vs FI median: **{si_p-fi_p:+.1f} pp** (lower is better).",
        f"Outsourcing delta vs FI median: **{si_o-fi_o:+.1f} pp** (lower is better).",
        "Interpretation: high outsourcing can be strategic (capacity flexibility, access to niche skills), but typically dilutes margin control. Use this view to test whether the mix aligns with margin outcomes.",
    ]
    section_ai_insight("Cost structure", bullets)

with tabs[1]:
    st.markdown("### Profitability drivers radar")
    st.plotly_chart(build_radar(df_snap_f), use_container_width=True)
    bullets = [
        "Billable & seniority are typical leading indicators for delivery margin.",
        "Offshore and outsourcing are levers for cost and scalability; the best outcome is usually achieved with strong governance and repeatable delivery models.",
        "Use the drill-down selector to compare a single peer against Siili in the 2x2 view."
    ]
    section_ai_insight("Drivers", bullets)

with tabs[2]:
    st.markdown("### Strategic positioning (2x2)")
    st.plotly_chart(build_2x2(df_snap_f), use_container_width=True)
    bullets = [
        "Upper-right (high RPE + high EBITDA) is a common ‘quality’ zone: pricing power + efficient delivery.",
        "If a company sits right but low, the ‘story’ is often cost inflation, utilization drag, or suboptimal delivery mix.",
        "Bubble size = revenue → helps contextualize scale vs productivity."
    ]
    section_ai_insight("2x2 positioning", bullets)

with tabs[3]:
    st.markdown("### Analyst expectations & forecasts (Siili)")
    st.caption("This section is a compact ‘expectations’ view. Update `data/analyst_estimates.csv` as new reports are published.")
    df_a = df_analyst.copy()
    # line chart for revenue and EBITA%
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.line(df_a, x="year", y="revenue_meur", color="source", markers=True)
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Revenue (€m)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(df_a, x="year", y="ebita_pct", color="source", markers=True)
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="EBITA (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_a.sort_values(["source","year"]), use_container_width=True, hide_index=True)

    bullets = [
        "Use this view to highlight ‘expectations gap’: if the market assumes a faster recovery than internal plan, IR messaging and execution cadence matter more.",
        "When estimates differ materially between houses, focus discussion on the driver: demand recovery timing, utilization, wage inflation, and mix (international/public/managed services)."
    ]
    section_ai_insight("Analyst view", bullets)

with tabs[4]:
    st.markdown("### Strategy & market map (best-effort)")
    st.caption("Strategic offering and delivery model mapping. Classifications are best-effort and meant as a starting point.")
    d = df_offering.merge(df_snap_all[df_snap_all["year"]==snapshot_year][["company","revenue_meur","ebitda_pct"]], on="company", how="left")
    fig = px.scatter(
        d,
        x="ebitda_pct",
        y="revenue_meur",
        color="country_group",
        size="revenue_meur",
        hover_name="company",
        hover_data={"service_focus":True,"delivery_model":True,"industry_focus":True,"scale_bucket":True,"revenue_meur":":.1f","ebitda_pct":":.1f"},
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="EBITDA (%)", yaxis_title="Revenue (€m)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Market offering (summary table)")
    st.dataframe(d[["company","country_group","service_focus","delivery_model","industry_focus","scale_bucket"]], use_container_width=True, hide_index=True)

    # Strategic narrative (rule-based)
    bullets = [
        "In Finland, mid-cap IT services often compete on (1) public sector frames, (2) design+engineering talent, and (3) delivery capacity. International peers add scale and industrialized delivery models.",
        "A differentiator to highlight for Siili: cross-border delivery + AI/data acceleration, while maintaining senior engineering credibility.",
        "If Siili’s EBITDA is below peers in the same ‘scale bucket’, typical strategic actions are: sharpen account focus, productize recurring services, and reduce bench via tighter demand shaping."
    ]
    section_ai_insight("Strategic narrative", bullets)

with tabs[5]:
    st.markdown("### Data & export")
    st.write("Download the datasets used by this dashboard:")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("Download benchmark_snapshot.csv", data=df_snap_all.to_csv(index=False), file_name="benchmark_snapshot.csv", mime="text/csv")
    with c2:
        st.download_button("Download benchmark_trends.csv", data=df_trends_all.to_csv(index=False), file_name="benchmark_trends.csv", mime="text/csv")
    with c3:
        st.download_button("Download analyst_estimates.csv", data=df_analyst.to_csv(index=False), file_name="analyst_estimates.csv", mime="text/csv")

    st.markdown("#### Notes")
    st.info("This repo is built to be easy to extend: add companies/years in the CSVs and the visuals update automatically.")
    st.code("Schema reference: see schema.md", language="text")
