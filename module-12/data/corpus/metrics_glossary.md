# Metrics glossary

- **growth**: period-over-period change of a metric, `(current - previous) / previous`,
  reported as a percentage. Quarterly growth compares a quarter to the previous quarter.
- **MoM / QoQ**: month-over-month / quarter-over-quarter change. Default reporting cadence is QoQ.
- **ARPU**: average revenue per unit = `net_rev / units` for a segment. Free-tier ARPU is 0
  because Free rows always have `net_rev = 0` (see the revenue policy).
- **retention**: `1 - churn_flag` for a segment. Because `churn_flag` is a rate (0–1), retention
  is also a fraction, not a count.
