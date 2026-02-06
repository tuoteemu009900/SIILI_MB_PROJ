# Data schema (v5)

All dataset columns are **snake_case** and stable for downstream usage.

## data/benchmark_snapshot.csv (one row per company per snapshot year)
- `year` (int)
- `company` (str)
- `country_group` (Finland | International | Target)
- `is_target` (0/1)
- `revenue_meur` (float) – revenue in €m
- `ebitda_pct` (float) – EBITDA margin (%)
- `personnel_cost_pct` (float) – personnel costs as % of revenue
- `outsourcing_pct` (float) – outsourcing/subcontracting as % of revenue
- `billable_pct` (float)
- `senior_pct` (float)
- `offshore_pct` (float)
- `headcount` (int)
- `revenue_per_employee_keur` (float) – derived: revenue/headcount in k€
- `service_focus` (str) – strategic offering summary
- `estimated_imputed` (0/1) – whether values are estimated / placeholder
- `source_note` (str)

## data/benchmark_trends.csv (multi-year, used for trend charts)
Same as snapshot but without `revenue_per_employee_keur` and `source_note`.

## data/analyst_estimates.csv (Siili only)
- `source` (Inderes | SEB)
- `report_date` (YYYY-MM-DD)
- `year` (int)
- `revenue_meur` (float)
- `ebita_meur` (float)
- `ebita_pct` (float)
- `eps_adj` (float)
- `dividend` (float)

## data/strategic_offering.csv
- `company`, `country_group`, `service_focus`, `delivery_model`, `industry_focus`, `scale_bucket`
