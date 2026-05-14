import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
cust = pd.read_csv("../Data/customer.csv", parse_dates=["order_date", "payment_date", "acquisition_date"])
ship = pd.read_csv("../Data/shipment.csv", parse_dates=["date"])
log  = pd.read_csv("../Data/logistics_performance.csv", parse_dates=["date"])

ship["customs_clearance_time_days"] = pd.to_numeric(ship["customs_clearance_time_days"], errors="coerce")
ship["O_Country"] = ship["O_Country"].str.strip()
ship["D_Country"] = ship["D_Country"].str.strip()

# ── Country → Region map ──────────────────────────────────────────────────────
country_to_region = {
    "USA": "North America", "Canada": "North America", "Mexico": "North America",
    "UK": "Europe", "Germany": "Europe", "France": "Europe", "Netherlands": "Europe",
    "India": "Asia-Pacific", "China": "Asia-Pacific", "Japan": "Asia-Pacific",
    "Australia": "Asia-Pacific", "Singapore": "Asia-Pacific",
}
ship["region"] = ship["D_Country"].map(country_to_region).fillna("Other")

# ── Aggregate logistics_performance to monthly level ──────────────────────────
log["year_month"] = log["date"].dt.to_period("M")
log_monthly = (
    log.groupby(["year_month", "region"])
    .agg(
        avg_delay_hours        = ("delay_hours_avg",              "mean"),
        avg_fuel_price         = ("fuel_price_usd_per_barrel",    "mean"),
        avg_warehouse_util     = ("warehouse_utilization_percent", "mean"),
        avg_damage_claims      = ("damage_claims_count",           "mean"),
    )
    .reset_index()
)

# ── Aggregate customer.csv by market_segment (acts as region proxy) ───────────
cust["days_to_pay"] = (cust["payment_date"] - cust["order_date"]).dt.days

cust_agg = (
    cust.groupby("market_segment")
    .agg(
        avg_lead_time_days     = ("lead_time_days",     "mean"),
        avg_satisfaction       = ("satisfaction_score", "mean"),
        avg_support_tickets    = ("support_tickets",    "mean"),
        avg_order_value        = ("order_value_usd",    "mean"),
        avg_days_to_pay        = ("days_to_pay",        "mean"),
    )
    .reset_index()
    .rename(columns={"market_segment": "region"})
)

# ── Supplier-level delay history from customer ────────────────────────────────
supplier_stats = (
    cust.groupby("supplier_id")
    .agg(
        supplier_avg_lead_time = ("lead_time_days",     "mean"),
        supplier_avg_sat       = ("satisfaction_score", "mean"),
        supplier_total_orders  = ("order_id",           "count"),
    )
    .reset_index()
)

# ── Base: shipment.csv (has the target: delivery_status) ─────────────────────
df = ship.dropna(subset=["delivery_status"]).copy()
df["year_month"] = df["date"].dt.to_period("M")

# ── Join logistics monthly stats ──────────────────────────────────────────────
df = df.merge(log_monthly, on=["year_month", "region"], how="left")

# ── Join customer aggregated stats ────────────────────────────────────────────
df = df.merge(cust_agg, on="region", how="left")

# ── Engineered features ───────────────────────────────────────────────────────
df["freight_ratio"]     = df["freight_cost"] / df["value"].replace(0, np.nan)
df["is_import"]         = (df["type"] == "Import").astype(int)
df["month"]             = df["date"].dt.month
df["quarter"]           = df["date"].dt.quarter

# ── Target encode ─────────────────────────────────────────────────────────────
df["target"] = (df["delivery_status"] == "Delayed").astype(int)

# ── Select final columns ──────────────────────────────────────────────────────
features = [
    # identifiers (keep for traceability, not used in model)
    "shipment_id", "date",

    # raw shipment features
    "is_import",
    "product_category",
    "O_Country", "D_Country", "region",
    "value",
    "freight_cost",
    "freight_ratio",
    "customs_clearance_time_days",
    "month", "quarter",

    # logistics performance features (joined)
    "avg_delay_hours",
    "avg_fuel_price",
    "avg_warehouse_util",
    "avg_damage_claims",

    # customer/region-level features (joined)
    "avg_lead_time_days",
    "avg_satisfaction",
    "avg_support_tickets",
    "avg_order_value",
    "avg_days_to_pay",

    # target
    "target",
]

# ── Fill missing logistics cols with global means (sparse time coverage) ──────
log_global = {
    "avg_delay_hours":    log["delay_hours_avg"].mean(),
    "avg_fuel_price":     log["fuel_price_usd_per_barrel"].mean(),
    "avg_warehouse_util": log["warehouse_utilization_percent"].mean(),
    "avg_damage_claims":  log["damage_claims_count"].mean(),
}
for col, val in log_global.items():
    df[col] = df[col].fillna(val)

# ── Fill missing customer cols with global means ──────────────────────────────
for col in ["avg_lead_time_days", "avg_satisfaction", "avg_support_tickets",
            "avg_order_value", "avg_days_to_pay"]:
    df[col] = df[col].fillna(df[col].mean())

out = df[features].copy()

print(f"Final dataset shape: {out.shape}")
print(f"Target distribution:\n{out['target'].value_counts()}")
print(f"Missing values:\n{out.isnull().sum()[out.isnull().sum() > 0]}")

out.to_csv("model_features.csv", index=False)
print("\nSaved to filtered/model_features.csv")
