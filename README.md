# Market Benchmark Dashboard — v3 (CEO-ready)

Highlights:
- Direction-aware delta coloring (**green = good**, **red = bad**) across KPIs and tables
- Trend view with peer medians (Finland vs International) + Siili line
- Cost structure + drivers radar + 2x2 positioning with drill-down
- Data quality toggles: `reported`, `estimated`, `estimated_imputed`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
Edit `data/benchmark_long.csv`.

Notes:
- Some peers have partial multi-year public data; other rows are included as placeholders.
- Missing values may be filled as `estimated_imputed` (peer median by Region+Year).
