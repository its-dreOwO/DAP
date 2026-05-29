import pandas as pd

COUNTRY_TO_REGION = {
    "USA": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Germany": "Europe",
    "France": "Europe",
    "UK": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "China": "Asia",
    "Japan": "Asia",
    "India": "Asia",
    "South Korea": "Asia",
    "Brazil": "South America",
    "Argentina": "South America",
    "Australia": "Oceania",
    "South Africa": "Africa",
}


def load_sources():
    customer = pd.read_csv("../../Data/customer.csv")
    shipment = pd.read_csv("../../Data/shipment.csv")
    logistics = pd.read_csv("../../Data/logistics_performance.csv")
    return customer, shipment, logistics


def merge(customer, shipment, logistics):
    shipment["region"] = shipment["D_Country"].map(COUNTRY_TO_REGION)
    merged = shipment.merge(customer, on="supplier_id", how="inner")
    merged = merged.merge(logistics[["region", "delay_hours_avg", "warehouse_utilization_percent"]], on="region", how="inner")
    merged["is_delayed"] = (merged["delivery_status"].str.strip() == "Delayed").astype(int)
    merged = merged.dropna()
    return merged


def main():
    customer, shipment, logistics = load_sources()
    df = merge(customer, shipment, logistics)
    out = "model_features.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} cols → {out}")
    print("Delayed rate:", df["is_delayed"].mean().round(3))


if __name__ == "__main__":
    main()
