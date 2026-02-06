[README_DATA.md](https://github.com/user-attachments/files/25125247/README_DATA.md)
# Data workflow

- The dashboard reads `data/benchmark_long.csv`.
- Rows have `DataQuality`:
  - `reported` = sourced from public IR/annual report/press release
  - `estimated` = manual placeholder
  - `estimated_imputed` = filled with peer median (by Region+Year)

## How to update
1. Edit the CSV and replace placeholders with verified values.
2. Keep `SourceNote` short but traceable (URL + what you used).
3. If you remove imputation, set missing values to empty and turn off "Include imputed" in the app.
