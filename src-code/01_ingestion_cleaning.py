import pandas as pd


def load_data():
    customer = pd.read_csv("Data/customer.csv")
    shipment = pd.read_csv("Data/shipment.csv")
    logistics = pd.read_csv("Data/logistics_performance.csv")
    return customer, shipment, logistics


def clean_data(customer, shipment, logistics):
    shipment["delivery_status"] = shipment["delivery_status"].str.strip()
    customer = customer.dropna(subset=["supplier_id", "lead_time_days"])
    logistics = logistics.dropna()
    return customer, shipment, logistics


if __name__ == "__main__":
    customer, shipment, logistics = load_data()
    customer, shipment, logistics = clean_data(customer, shipment, logistics)
    print(
        "Rows — customer:",
        len(customer),
        "shipment:",
        len(shipment),
        "logistics:",
        len(logistics),
    )
