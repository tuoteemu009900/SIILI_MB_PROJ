[README.md](https://github.com/user-attachments/files/25126059/README.md)
# Siili Market Benchmark Dashboard (v5)

A CEO-ready Streamlit dashboard for benchmarking Siili Solutions vs Finnish and international peers.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub
2. In Streamlit Cloud: **New app** → select repo → main file: `app.py`
3. Ensure the following paths exist in the repo:
   - `data/`
   - `.streamlit/`

## Update data
- Snapshot: `data/benchmark_snapshot.csv`
- Trends: `data/benchmark_trends.csv`
- Analyst estimates: `data/analyst_estimates.csv`

See `schema.md` for stable field names.
