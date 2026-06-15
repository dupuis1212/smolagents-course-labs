# Revenue recognition policy

`net_rev` recognizes revenue in the month the subscription is active, after deducting
refunds issued in the same month. Free-tier rows always have `net_rev = 0`. When comparing
growth across tiers, use `net_rev` (not `units`), because price differs by tier.
