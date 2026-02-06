# Data sources & notes (for traceability)

This dashboard uses a mix of:
- company investor pages / official PDFs
- press releases / financial statements bulletins
- (some metrics) computed from publicly stated values

## Siili Solutions (FY2024)
- Revenue, EBITDA %, Headcount: Siili “Key figures” page.

## Gofore (FY2024)
- Net sales and Adjusted EBITA-% + headcount from financial statements release (STT Info).

## Digital Workforce Services (FY2024)
- Net sales, EBITDA% and headcount from the Financial Statements Bulletin 2024 (PDF).

## Solteq (FY2024)
- Revenue, EBITDA and headcount from Financial Statements Bulletin 2024 (PDF).

## Sopra Steria (FY2024)
- Operating margin % and operating margin amount from FY2024 result announcement.
- Revenue is inferred as: Revenue = operating_margin_amount / operating_margin_%.
- Headcount from FY2024 results page (“50,000 employees”).

## Reply (FY2024)
- Revenue, gross operating income (used as EBITDA proxy), and employees from Annual Financial Report highlights.

## Endava (FY2024)
- Revenue (GBP) and headcount from FY2024 results press release.
- Revenue converted to EUR using average GBP/EUR 2024 (external FX-history source).

## Alten (FY2024)
- Revenue and “operating margin on activity %” from annual results page (used as EBITDA proxy).

## Cegeka (FY2024)
- Revenue and operating profit from Cegeka annual report interview page.
- EBITDA% uses operating profit / revenue as a proxy.

## Missing KPIs
PersonnelCost_pct, Outsourcing_pct, Billable_pct, Senior_pct, Offshore_pct:
- left empty intentionally; add values when available (annual report notes, investor presentations, or consistent estimates).
