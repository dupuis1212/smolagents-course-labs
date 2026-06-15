# Data dictionary — sales.csv

- **month**: reporting month, `YYYY-MM`.
- **region_code**: sales region. `NA` = North America, `EU` = Europe, `APAC` = Asia-Pacific.
- **category**: subscription tier. One of `Free`, `Pro`, `Team`.
- **units**: number of active subscriptions sold in that month/region/category.
- **net_rev**: net revenue in USD = gross subscription revenue **minus refunds**. It is NOT
  gross revenue; refunds are already subtracted.
- **churn_flag**: monthly churn **rate** (a fraction 0–1) for that segment, NOT a 0/1 flag.
  Despite the name, higher means worse retention.
