# Customer segmentation rules

A *segment* is a unique `(region_code, category)` pair in a given `month`. When a question asks
about a "region" use `region_code` (`NA`, `EU`, `APAC`); when it asks about a "tier" or "plan"
use `category` (`Free`, `Pro`, `Team`).

Free-tier (`category = Free`) rows carry users but no revenue (`net_rev = 0`), so exclude them
when ranking segments by `net_rev`. Include them when counting `units` or active subscriptions.
