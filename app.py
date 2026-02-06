\
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Siili Market Benchmark", layout="wide", page_icon="📊")

@st.cache_data
def load_data():
    df = pd.read_csv("data/benchmark_data.csv")
    # Ensure numeric columns
    num_cols = ["Revenue_EURm","EBITDA_pct","PersonnelCost_pct","Outsourcing_pct",
                "Headcount","Billable_pct","Senior_pct","Offshore_pct","Revenue_per_Employee_EUR"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("Filters")
group_choice = st.sidebar.radio("Peers", ["All", "Finland", "International"], index=0)
company_choice = st.sidebar.selectbox("Drill-down company", ["(None)"] + sorted(df["Company"].unique().tolist()))

filtered = df.copy()
if group_choice != "All":
    filtered = filtered[filtered["Group"] == group_choice]

# Always keep Siili visible in comparisons even if Finland/International filter selected
siili = df[df["Company"] == "Siili Solutions"]
if not siili.empty and "Siili Solutions" not in filtered["Company"].values:
    filtered = pd.concat([filtered, siili], ignore_index=True)

# Helpers
def median_for(group_name, col):
    sub = df[(df["Role"] == "Peer") & (df["Group"] == group_name)]
    return float(sub[col].median(skipna=True)) if col in df.columns else np.nan

def format_pct(x):
    return "—" if pd.isna(x) else f"{x:.1f}%"

def format_eurm(x):
    return "—" if pd.isna(x) else f"€{x:,.1f}M"

def format_eur(x):
    return "—" if pd.isna(x) else f"€{x:,.0f}"

def gauge(title, value, median_fi, median_int, suffix="%", rng=(0, 30)):
    fig = go.Figure()
    # main gauge shows Siili
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value if not pd.isna(value) else 0,
        number={"suffix": suffix},
        title={"text": title},
        gauge={
            "axis": {"range": list(rng)},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [rng[0], rng[1]], "color": "rgba(0,0,0,0.05)"}
            ],
            # threshold lines for medians
            "threshold": {"line": {"width": 2}, "value": value if not pd.isna(value) else 0}
        }
    ))
    # Add median markers as shapes
    shapes = []
    for m, label in [(median_fi, "FI median"), (median_int, "INT median")]:
        if not pd.isna(m):
            shapes.append(dict(
                type="line",
                x0=0.15, x1=0.85,
                y0=m, y1=m,
                xref="paper", yref="y",
                line=dict(width=2, dash="dot")
            ))
    # Plotly indicator doesn't support yref shapes nicely; show medians in subtitle instead
    fig.update_layout(
        height=220,
        margin=dict(l=10,r=10,t=40,b=10),
        annotations=[
            dict(
                x=0.5, y=-0.15, xref="paper", yref="paper", showarrow=False,
                text=f"FI median: {format_pct(median_fi)} • INT median: {format_pct(median_int)}",
                font=dict(size=12)
            )
        ]
    )
    return fig

st.title("📊 Market Benchmark Dashboard — Siili vs Peers")

# Executive KPI Summary
siili_row = df[df["Company"] == "Siili Solutions"].iloc[0]

fi_median_ebitda = median_for("Finland","EBITDA_pct")
int_median_ebitda = median_for("International","EBITDA_pct")

fi_median_pers = median_for("Finland","PersonnelCost_pct")
int_median_pers = median_for("International","PersonnelCost_pct")

fi_median_rpe = median_for("Finland","Revenue_per_Employee_EUR")
int_median_rpe = median_for("International","Revenue_per_Employee_EUR")

fi_median_outs = median_for("Finland","Outsourcing_pct")
int_median_outs = median_for("International","Outsourcing_pct")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.plotly_chart(
        gauge("EBITDA", siili_row["EBITDA_pct"], fi_median_ebitda, int_median_ebitda, suffix="%", rng=(0, 25)),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with kpi_cols[1]:
    if pd.isna(siili_row["PersonnelCost_pct"]):
        st.info("PersonnelCost_% missing in CSV (ready to fill).")
    else:
        st.plotly_chart(
            gauge("Personnel Costs", siili_row["PersonnelCost_pct"], fi_median_pers, int_median_pers, suffix="%", rng=(0, 80)),
            use_container_width=True,
            config={"displayModeBar": False},
        )
with kpi_cols[2]:
    # Revenue per employee gauge (EUR)
    val = siili_row["Revenue_per_Employee_EUR"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val if not pd.isna(val) else 0,
        number={"prefix":"€", "valueformat":",.0f"},
        title={"text":"Revenue / Employee"},
        gauge={"axis":{"range":[0, 250000]}, "bar":{"thickness":0.25}}
    ))
    fig.update_layout(height=220, margin=dict(l=10,r=10,t=40,b=10),
                      annotations=[dict(x=0.5,y=-0.15,xref="paper",yref="paper",showarrow=False,
                                        text=f"FI median: {format_eur(fi_median_rpe)} • INT median: {format_eur(int_median_rpe)}",
                                        font=dict(size=12))])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with kpi_cols[3]:
    if pd.isna(siili_row["Outsourcing_pct"]):
        st.info("Outsourcing_% missing in CSV (ready to fill).")
    else:
        st.plotly_chart(
            gauge("Outsourcing", siili_row["Outsourcing_pct"], fi_median_outs, int_median_outs, suffix="%", rng=(0, 80)),
            use_container_width=True,
            config={"displayModeBar": False},
        )

st.divider()

# Main charts
left, right = st.columns([1.1, 1])

# Cost Structure
with left:
    st.subheader("B. Cost Structure")
    if filtered[["PersonnelCost_pct","Outsourcing_pct"]].dropna(how="all").empty:
        st.warning("PersonnelCost_% / Outsourcing_% not populated yet. Add values to enable the stacked bar.")
    else:
        cost_df = filtered[["Company","PersonnelCost_pct","Outsourcing_pct","Revenue_EURm"]].copy()
        cost_df = cost_df.dropna(subset=["PersonnelCost_pct","Outsourcing_pct"], how="all")
        cost_long = cost_df.melt(id_vars=["Company","Revenue_EURm"], value_vars=["PersonnelCost_pct","Outsourcing_pct"],
                                 var_name="CostType", value_name="Pct")
        fig = px.bar(cost_long, x="Company", y="Pct", color="CostType", barmode="stack",
                     hover_data={"Revenue_EURm":":.1f", "Pct":":.1f"})
        fig.update_layout(height=420, legend_title_text="", xaxis_title="", yaxis_title="% of revenue")
        st.plotly_chart(fig, use_container_width=True)

# Profitability Drivers
with right:
    st.subheader("C. Profitability Drivers")
    cols = ["Billable_pct","Senior_pct","Outsourcing_pct","Offshore_pct"]
    available = filtered[["Company"] + cols].dropna(subset=cols, how="all")
    if available.empty:
        st.warning("Driver KPIs not populated yet (Billable/Senior/Outsourcing/Offshore). Add values to enable radar chart.")
    else:
        radar_df = available.set_index("Company")[cols]
        categories = ["Billable","Senior","Outsourcing","Offshore"]
        fig = go.Figure()
        for company, row in radar_df.iterrows():
            values = [row[c] if not pd.isna(row[c]) else 0 for c in cols]
            fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                          fill="toself", name=company))
        fig.update_layout(height=420, polar=dict(radialaxis=dict(visible=True, range=[0, 100])), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Strategic Positioning 2x2
st.subheader("D. Strategic Positioning (2x2)")
pos = filtered.copy()
pos["Revenue_per_Employee_EUR"] = pos["Revenue_EURm"]*1e6 / pos["Headcount"]
pos = pos.dropna(subset=["Revenue_per_Employee_EUR","EBITDA_pct"], how="any")
if pos.empty:
    st.warning("Need both Headcount and EBITDA% to plot 2x2. Fill missing values in CSV.")
else:
    pos["Size"] = pos["Revenue_EURm"].fillna(0)
    color_map = {"Finland":"#777777", "International":"#999999", "Target":"#1f77b4"}
    pos["LegendGroup"] = pos.apply(lambda r: "Siili" if r["Company"]=="Siili Solutions" else r["Group"], axis=1)
    fig = px.scatter(
        pos,
        x="Revenue_per_Employee_EUR",
        y="EBITDA_pct",
        size="Size",
        color="LegendGroup",
        hover_name="Company",
        hover_data={
            "Revenue_EURm":":.1f",
            "Headcount":":.0f",
            "Revenue_per_Employee_EUR":":.0f",
            "EBITDA_pct":":.1f"
        },
    )
    fig.update_layout(height=520, xaxis_title="Revenue per Employee (€)", yaxis_title="EBITDA (%)")
    st.plotly_chart(fig, use_container_width=True)

# Drill-down section
st.subheader("Company Drill-down")
if company_choice != "(None)":
    r = df[df["Company"] == company_choice].iloc[0]
    st.markdown(f"### {company_choice}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue (EURm)", "—" if pd.isna(r["Revenue_EURm"]) else f"{r['Revenue_EURm']:.1f}")
    c2.metric("EBITDA %", "—" if pd.isna(r["EBITDA_pct"]) else f"{r['EBITDA_pct']:.1f}%")
    c3.metric("Headcount", "—" if pd.isna(r["Headcount"]) else f"{int(r['Headcount'])}")
    c4.metric("Revenue/Employee", "—" if pd.isna(r["Revenue_per_Employee_EUR"]) else f"€{r['Revenue_per_Employee_EUR']:.0f}")
    st.dataframe(r.to_frame("value"), use_container_width=True)

st.divider()

# AI Insights / Summary (simple rule-based narrative; easy to replace with LLM later)
st.subheader("E. Insights (auto-generated)")
ins = []
siili_ebitda = siili_row["EBITDA_pct"]
siili_rpe = siili_row["Revenue_per_Employee_EUR"]

if not pd.isna(siili_ebitda) and not pd.isna(fi_median_ebitda):
    delta = siili_ebitda - fi_median_ebitda
    ins.append(f"- **EBITDA**: Siili ({siili_ebitda:.1f}%) is {'above' if delta>0 else 'below'} Finland peer median ({fi_median_ebitda:.1f}%) by {delta:+.1f} pp.")
if not pd.isna(siili_ebitda) and not pd.isna(int_median_ebitda):
    delta = siili_ebitda - int_median_ebitda
    ins.append(f"- **EBITDA vs international**: {'above' if delta>0 else 'below'} international median ({int_median_ebitda:.1f}%) by {delta:+.1f} pp.")

if not pd.isna(siili_rpe) and not pd.isna(fi_median_rpe):
    delta = siili_rpe - fi_median_rpe
    ins.append(f"- **Revenue per employee**: Siili ({siili_rpe:,.0f}€) vs Finland median ({fi_median_rpe:,.0f}€): {delta:+,.0f}€.")
if not pd.isna(siili_rpe) and not pd.isna(int_median_rpe):
    delta = siili_rpe - int_median_rpe
    ins.append(f"- **Revenue per employee vs international**: {delta:+,.0f}€ vs international median ({int_median_rpe:,.0f}€).")

if not ins:
    st.write("Add more KPI data to enable richer auto-insights.")
else:
    st.markdown("\n".join(ins))

st.caption("Tip: Plotly charts include a built-in camera icon (modebar) for PNG export in most browsers.")
