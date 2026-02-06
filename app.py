from __future__ import annotations

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

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "benchmark_long.csv"

TARGET = "Siili Solutions"

GOOD_GREEN = "#22c55e"
BAD_RED = "#ef4444"
MUTED = "rgba(242,246,255,0.75)"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    num_cols = ["Revenue_EURm","EBITDA_pct","PersonnelCost_pct","Outsourcing_pct","Headcount",
                "Billable_pct","Senior_pct","Offshore_pct","RevenuePerEmployee_EURk"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def css():
    st.markdown("""
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
      .subtle { color: rgba(242,246,255,0.7); }
      .insight li { margin-bottom: 0.35rem; }
      .kpi-help { color: rgba(242,246,255,0.65); font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)


@dataclass
class Metric:
    key: str
    title: str
    suffix: str
    fmt: str
    higher_is_better: bool

from dataclasses import dataclass


METRICS = [
    Metric("EBITDA_pct", "EBITDA", "%", ".1f", True),
    Metric("PersonnelCost_pct", "Personnel", "%", ".1f", False),
    Metric("RevenuePerEmployee_EURk", "Revenue / Employee", "k€", ".0f", True),
    Metric("Outsourcing_pct", "Outsourcing", "%", ".1f", False),
]


def median(df: pd.DataFrame, metric: str, region: str, year: int) -> float | None:
    s = df[(df["Region"] == region) & (df["Year"] == year)][metric].dropna()
    if s.empty:
        return None
    return float(s.median())


def indicator(metric: Metric, value: float | None, ref: float | None) -> go.Figure:
    if value is None or np.isnan(value):
        value = 0.0
        show_delta = False
    else:
        show_delta = ref is not None and (not np.isnan(ref))

    # delta polarity: if lower is better -> invert "increasing/decreasing" colors
    if metric.higher_is_better:
        inc_color, dec_color = GOOD_GREEN, BAD_RED
    else:
        inc_color, dec_color = BAD_RED, GOOD_GREEN

    fig = go.Figure(go.Indicator(
        mode="number+delta" if show_delta else "number",
        value=float(value),
        number={"suffix": metric.suffix, "valueformat": metric.fmt},
        delta={
            "reference": float(ref),
            "valueformat": metric.fmt,
            "suffix": metric.suffix,
            "increasing": {"color": inc_color},
            "decreasing": {"color": dec_color},
        } if show_delta else None,
        title={"text": metric.title},
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=10,r=10,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
    )
    return fig


def make_trend(df: pd.DataFrame, metric: str, companies: list[str], regions: list[str]) -> go.Figure:
    d = df[df["Region"].isin(regions)].copy()
    fig = go.Figure()

    # peer medians over time
    for reg in regions:
        med = d[d["Region"] == reg].groupby("Year")[metric].median().dropna()
        if not med.empty:
            fig.add_trace(go.Scatter(
                x=med.index, y=med.values, mode="lines",
                name=f"{reg} median",
                line=dict(dash="dot"),
                hovertemplate="%{x}<br>Median: %{y:.2f}<extra></extra>"
            ))

    # selected companies
    for c in companies:
        s = df[df["Company"] == c].sort_values("Year")
        s = s[["Year", metric]].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["Year"], y=s[metric], mode="lines+markers",
            name=c,
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
            opacity=1.0 if c == TARGET else 0.55
        ))

    fig.update_layout(
        height=320,
        margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Year",
        yaxis_title=metric,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0)
    )
    return fig


def section_insight_kpis(snapshot: pd.DataFrame, year: int) -> list[str]:
    out = []
    si = snapshot[snapshot["Company"] == TARGET]
    if si.empty:
        return ["Siili row missing for selected year."]
    si = si.iloc[0]

    fin = median(snapshot, "EBITDA_pct", "Finland (Peer)", year)
    intl = median(snapshot, "EBITDA_pct", "International (Peer)", year)
    e = si.get("EBITDA_pct", np.nan)

    if pd.notna(e) and fin is not None:
        out.append(f"**EBITDA-%:** Siili {e:.1f}% vs FI median {fin:.1f}% → {e-fin:+.1f} pp.")
    if pd.notna(e) and intl is not None:
        out.append(f"**EBITDA-%:** Siili {e:.1f}% vs INT median {intl:.1f}% → {e-intl:+.1f} pp.")

    rpe = si.get("RevenuePerEmployee_EURk", np.nan)
    fin_rpe = median(snapshot, "RevenuePerEmployee_EURk", "Finland (Peer)", year)
    intl_rpe = median(snapshot, "RevenuePerEmployee_EURk", "International (Peer)", year)

    if pd.notna(rpe) and fin_rpe is not None:
        out.append(f"**Revenue/Employee:** Siili {rpe:.0f}k€ vs FI median {fin_rpe:.0f}k€ → {rpe-fin_rpe:+.0f}k€.")
    if pd.notna(rpe) and intl_rpe is not None:
        out.append(f"**Revenue/Employee:** Siili {rpe:.0f}k€ vs INT median {intl_rpe:.0f}k€ → {rpe-intl_rpe:+.0f}k€.")

    # data quality note
    imputed = (snapshot["DataQuality"] == "estimated_imputed").mean() * 100
    if imputed > 0:
        out.append(f"**Data quality:** {imputed:.0f}% riveistä on *estimated_imputed* (täytetty peer-medianilla).")
    return out


def build_cost(snapshot: pd.DataFrame) -> go.Figure:
    d = snapshot.sort_values(["Region","Company"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["Company"], y=d["PersonnelCost_pct"],
        name="Personnel (% of revenue)",
        hovertemplate="<b>%{x}</b><br>Personnel: %{y:.1f}%<extra></extra>",
        opacity=np.where(d["Company"] == TARGET, 1.0, 0.85),
    ))
    fig.add_trace(go.Bar(
        x=d["Company"], y=d["Outsourcing_pct"],
        name="Outsourcing (% of revenue)",
        hovertemplate="<b>%{x}</b><br>Outsourcing: %{y:.1f}%<extra></extra>",
        opacity=np.where(d["Company"] == TARGET, 1.0, 0.85),
    ))
    fig.update_layout(
        barmode="stack",
        height=420,
        xaxis=dict(tickangle=-25),
        yaxis=dict(title="% of revenue", rangemode="tozero"),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
    )
    return fig


def build_radar(snapshot: pd.DataFrame, companies: list[str]) -> go.Figure:
    dims = [("Billable_pct","Billable %"),("Senior_pct","Senior %"),("Outsourcing_pct","Outsourcing %"),("Offshore_pct","Offshore %")]
    fig = go.Figure()
    for c in companies:
        r = snapshot[snapshot["Company"] == c]
        if r.empty:
            continue
        r = r.iloc[0]
        vals = [r.get(k, np.nan) for k,_ in dims]
        if all(pd.isna(v) for v in vals):
            continue
        vals = [0 if pd.isna(v) else float(v) for v in vals]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[lab for _,lab in dims] + [dims[0][1]],
            fill="toself" if c == TARGET else "none",
            name=c,
            opacity=0.95 if c == TARGET else 0.55,
        ))
    fig.update_layout(
        height=420,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0)
    )
    return fig


def build_scatter(snapshot: pd.DataFrame) -> go.Figure:
    d = snapshot.copy()
    d["Group"] = d["Region"].map({"Finland (Peer)":"Finland", "International (Peer)":"International"}).fillna(d["Region"])
    d["Size"] = d["Revenue_EURm"].fillna(0)

    fig = px.scatter(
        d, x="RevenuePerEmployee_EURk", y="EBITDA_pct", size="Size", color="Group",
        hover_name="Company",
        hover_data={
            "Revenue_EURm":":.1f",
            "Headcount":True,
            "PersonnelCost_pct":":.1f",
            "Outsourcing_pct":":.1f",
            "Billable_pct":":.1f",
            "Senior_pct":":.1f",
            "Offshore_pct":":.1f",
            "DataQuality":True,
        },
        height=540,
    )
    fig.update_traces(marker=dict(line=dict(width=np.where(d["Company"]==TARGET, 2.5, 1.0), color="rgba(255,255,255,0.25)")))
    fig.update_layout(
        xaxis_title="Revenue per Employee (k€)",
        yaxis_title="EBITDA (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
    )
    return fig


def color_delta(val: float) -> str:
    if pd.isna(val):
        return ""
    return f"color: {GOOD_GREEN}; font-weight: 700;" if val >= 0 else f"color: {BAD_RED}; font-weight: 700;"


def main():
    st.set_page_config(page_title="Market Benchmark — Siili", layout="wide", initial_sidebar_state="expanded")
    css()
    df = load_data()

    years = sorted(df["Year"].unique().tolist())
    default_year = max(years)

    with st.sidebar:
        st.markdown("### Controls")
        year = st.select_slider("Snapshot year", options=years, value=default_year)
        regions = st.multiselect("Peer groups", ["Finland (Peer)", "International (Peer)"], default=["Finland (Peer)", "International (Peer)"])
        include_imputed = st.toggle("Include estimated/imputed", value=True)
        st.markdown("---")
        trend_metric = st.selectbox("Trend metric", ["Revenue_EURm","EBITDA_pct","RevenuePerEmployee_EURk"], index=1)

    snap = df[(df["Year"] == year) & (df["Region"].isin(regions))].copy()
    if not include_imputed:
        snap = snap[snap["DataQuality"] == "reported"].copy()

    # ensure Siili present
    if TARGET not in snap["Company"].unique():
        t = df[(df["Year"] == year) & (df["Company"] == TARGET)]
        if not t.empty:
            snap = pd.concat([snap, t], ignore_index=True)

    # Header
    st.markdown(f"""
    <div class="sticky">
      <h1 style="margin:0;">Market Benchmark <span class="badge">{TARGET}</span></h1>
      <div class="subtle">CEO-ready • deltas colored (green=good, red=bad) • trends + drill-down • {year}</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    kcols = st.columns(4, gap="large")
    for i, m in enumerate(METRICS):
        with kcols[i]:
            trow = snap[snap["Company"] == TARGET]
            val = None if trow.empty else trow.iloc[0].get(m.key, np.nan)
            val = None if pd.isna(val) else float(val)
            ref = median(snap, m.key, "Finland (Peer)", year)
            st.plotly_chart(indicator(m, val, ref), use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"<div class='kpi-help'>Delta vs Finland median (direction-aware).</div>", unsafe_allow_html=True)

    # Summary + Trend
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("### Executive insights")
        bullets = section_insight_kpis(snap, year)
        st.markdown("<div class='card insight'>", unsafe_allow_html=True)
        st.markdown("\n".join([f"- {b}" for b in bullets]))
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### Trends")
        # pick a set: Siili + FI median + INT median
        companies = [TARGET]
        fig = make_trend(df, trend_metric, companies, regions)
        fig.update_yaxes(title=trend_metric)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Cost Structure", "Drivers Radar", "Strategic Positioning (2x2)", "Data & Export"])

    with tab1:
        c1, c2 = st.columns([1.55, 1], gap="large")
        with c1:
            st.markdown("#### Cost structure (% of revenue)")
            st.plotly_chart(build_cost(snap), use_container_width=True)
        with c2:
            st.markdown("#### Insights")
            st.markdown("<div class='card insight'>", unsafe_allow_html=True)
            st.markdown("- Jos **Personnel%** on korkea, pienet käyttöaste-muutokset vaikuttavat nopeasti.\n"
                        "- Jos **Outsourcing%** on korkea, kapasiteetti joustaa, mutta kate voi olla herkempi.\n"
                        "- Deltat ovat **direction-aware**: kulujen nousu näkyy punaisena.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Profitability drivers (radar)")
        companies_all = sorted(snap["Company"].unique().tolist())
        default = [TARGET] + [c for c in companies_all if c != TARGET][:7]
        sel = st.multiselect("Companies", companies_all, default=default)
        st.plotly_chart(build_radar(snap, sel[:12]), use_container_width=True)
        st.caption("Driver-metriikat voivat sisältää *estimated_imputed* arvoja (toggle sivupalkissa).")

    with tab3:
        c1, c2 = st.columns([1.55, 1], gap="large")
        with c1:
            fig = build_scatter(snap)
            selected = None
            if PLOTLY_EVENTS:
                pts = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=560)
                st.caption("Klikkaa pistettä drill-downiin (jos klikkaus ei toimi, varmista dependency `streamlit-plotly-events`).")
                if pts:
                    # safer: use pointNumber and match hover text
                    pn = pts[0].get("pointNumber")
                    if pn is not None and pn < len(fig.data[0]["x"]):
                        # fall back: use customdata not set; so provide selectbox too
                        pass
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Drill-down")
            selected = st.selectbox("Company", sorted(snap["Company"].unique().tolist()), index=0)
            r = snap[snap["Company"] == selected].iloc[0]
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{selected}**  \n"
                        f"Region: {r['Region']}  \n"
                        f"Revenue: {r['Revenue_EURm']:.1f} €m  \n"
                        f"EBITDA: {r['EBITDA_pct']:.1f}%  \n"
                        f"Headcount: {int(r['Headcount']) if pd.notna(r['Headcount']) else '—'}  \n"
                        f"Revenue/Employee: {r['RevenuePerEmployee_EURk']:.0f} k€  \n"
                        f"Data quality: **{r['DataQuality']}**")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("#### Snapshot table (with deltas vs FI median)")
        show_cols = ["Company","Region","Year","Revenue_EURm","EBITDA_pct","PersonnelCost_pct","Outsourcing_pct","Headcount","RevenuePerEmployee_EURk","DataQuality"]
        t = snap[show_cols].copy()

        # deltas vs FI median (direction-aware)
        for m in METRICS:
            ref = median(snap, m.key, "Finland (Peer)", year)
            if ref is None:
                t[f"Δ {m.title} vs FI median"] = np.nan
            else:
                t[f"Δ {m.title} vs FI median"] = t[m.key] - ref
                if not m.higher_is_better:
                    # invert sign so green always means "good"
                    t[f"Δ {m.title} vs FI median"] = -(t[m.key] - ref)

        delta_cols = [c for c in t.columns if c.startswith("Δ ")]
        sty = t.style.applymap(color_delta, subset=delta_cols).format({
            "Revenue_EURm":"{:.1f}",
            "EBITDA_pct":"{:.1f}",
            "PersonnelCost_pct":"{:.1f}",
            "Outsourcing_pct":"{:.1f}",
            "RevenuePerEmployee_EURk":"{:.0f}",
            **{c:"{:+.2f}" for c in delta_cols}
        })

        st.dataframe(sty, use_container_width=True, hide_index=True)

        st.download_button(
            "Download snapshot CSV",
            data=t.to_csv(index=False).encode("utf-8"),
            file_name=f"benchmark_snapshot_{year}.csv",
            mime="text/csv",
        )

        st.markdown("#### Data notes")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("- **Green = good, Red = bad**, myös kulu-%:ssa (direction-aware).\n"
                    "- `estimated_imputed` = puuttuva arvo täytetty peer-medianilla (Region+Year).\n"
                    "- Korvaa arviot omilla varmennetuilla luvuilla `data/benchmark_long.csv`-tiedostossa.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
