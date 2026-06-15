import csv, pathlib, random
random.seed(42)
root = pathlib.Path("_assets")
# --- sales.csv : monthly sales by region/category, with deliberately ambiguous columns ---
months = [f"2025-{m:02d}" for m in range(1, 13)]
regions = ["NA", "EU", "APAC"]
cats = ["Free", "Pro", "Team"]
base = {"Free": 200, "Pro": 1200, "Team": 4000}
growth = {"Free": 1.01, "Pro": 1.06, "Team": 1.10}  # Team grows fastest YoY
rows = []
for ci, cat in enumerate(cats):
    for r in regions:
        for i, mo in enumerate(months):
            units = int(base[cat] * (growth[cat] ** i) * random.uniform(0.9, 1.1))
            price = {"Free": 0, "Pro": 29, "Team": 99}[cat]
            gross = units * price
            refunds = gross * random.uniform(0.0, 0.06)
            net_rev = round(gross - refunds, 2)
            churn = round(random.uniform(0.01, 0.09) + (0.03 if cat == "Free" else 0), 3)
            rows.append({"month": mo, "region_code": r, "category": cat,
                         "units": units, "net_rev": net_rev, "churn_flag": churn})
with (root / "sales.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["month", "region_code", "category", "units", "net_rev", "churn_flag"])
    w.writeheader(); w.writerows(rows)
# --- customers.csv : per-customer churn snapshot (Module 10 web-comparison question) ---
crows = []
for cid in range(1, 121):
    cat = random.choice(cats); r = random.choice(regions)
    crows.append({"customer_id": cid, "region_code": r, "plan": cat,
                  "mrr": {"Free": 0, "Pro": 29, "Team": 99}[cat],
                  "tenure_months": random.randint(1, 36),
                  "churned": 1 if random.random() < (0.22 if cat == "Free" else 0.08) else 0})
with (root / "customers.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["customer_id", "region_code", "plan", "mrr", "tenure_months", "churned"])
    w.writeheader(); w.writerows(crows)
print("sales.csv rows:", len(rows), "| customers.csv rows:", len(crows))
