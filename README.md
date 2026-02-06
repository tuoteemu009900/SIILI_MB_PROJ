# Siili Market Benchmark Dashboard (Streamlit + Plotly)

CEO-/johtotason selainpohjainen benchmark-dashboard Siili Solutions Oyj:n asemointiin kotimaisiin ja ulkomaisiin verrokkeihin nähden.

## Mitä saat
- **KPI-kortit** (Siili vs kotimainen mediaani vs ulkomainen mediaani)
- **Cost Structure** (placeholder: Personnel vs Outsourcing % — datakentät valmiina)
- **Profitability Drivers** (radar: Billable/Senior/Outsourcing/Offshore — datakentät valmiina)
- **Strategic Positioning 2x2** (Revenue/Employee vs EBITDA%)
- **AI Insights** (dynaaminen tiivistelmä datan mukaan)

> Huom: Kaikki KPI:t eivät ole julkisista lähteistä helposti saatavissa yhtenäisesti (esim. Billable/Senior/Offshore sekä usein Personnel/Outsourcing-%). CSV:ssä nämä on jätetty tyhjiksi (`NaN`) ja dashboard toimii silti, mutta tietyt näkymät näyttävät “data missing” -viestin kunnes täydennät arvot.

## Käyttöönotto (local)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data
- `data/benchmark_data.csv` sisältää valmiin rungon + osan luvuista valmiiksi.
- Lisää/korvaa yrityksiä riveinä, tai lisää uusia KPI-kenttiä.

## Deploy
- Streamlit Cloud: liitä GitHub-repo ja valitse `app.py`.
- Vaihtoehtoisesti Docker/VM.

## Lähteet
Katso `data_sources.md` (URL-listaus + selite).
