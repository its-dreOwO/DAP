# Báo cáo Cleaning — ALIGNED với contract nhóm

Hướng: **by Department** (theo thầy) · Nhãn: **Late_delivery_risk** (theo contract nhóm) · Giữ survival riêng.

## Các bước (khớp 01_ingestion_cleaning.py)

- Input rows x cols: 180,519 x 53
- Dropped LEAKAGE (7): ['Days for shipping (real)', 'Delivery Status', 'shipping date (DateOrders)', 'Order Status', 'Benefit per order', 'Order Profit Per Order', 'Order Item Profit Ratio']
- Dropped PII (6): ['Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname', 'Customer Street', 'Customer Zipcode']
- Dropped EMPTY/REDUNDANT (11): ['Product Description', 'Order Zipcode', 'Product Image', 'Product Status', 'Category Id', 'Department Id', 'Product Card Id', 'Product Category Id', 'Order Item Cardprod Id', 'Customer Id', 'Order Customer Id']
- Impossible-value rows removed: 0
- Duplicate rows removed: 0
- Output (classification) rows x cols: 180,519 x 33
- Unique orders (group key): 65,752
- Items per order: 2.75
- Target Late_delivery_risk -> late rate: 54.8% (0=81,542, 1=98,977)
- Saved classification dataset -> clean_data.csv
- Saved survival dataset (with duration) -> clean_data_survival.csv (180,519 rows)

## Bảng rủi ro theo Department (late rate)

| Department | N orders | Late rate | Avg scheduled days |
|---|---|---|---|
| Pet Shop | 492 | 58.9% | 2.68 |
| Book Shop | 405 | 56.5% | 2.86 |
| Health and Beauty  | 362 | 55.8% | 2.93 |
| Fitness | 2,479 | 55.5% | 2.92 |
| Outdoors | 9,686 | 55.5% | 2.95 |
| Technology | 1,465 | 55.0% | 2.86 |
| Fan Shop | 66,861 | 54.8% | 2.93 |
| Golf | 33,220 | 54.8% | 2.92 |
| Apparel | 48,998 | 54.7% | 2.93 |
| Footwear | 14,525 | 54.7% | 2.93 |
| Discs Shop | 2,026 | 54.4% | 3.01 |

## Lưu ý quan trọng
- File `clean_data.csv` KHÔNG chứa cột leakage -> dùng cho **classification/modeling**.
- File `clean_data_survival.csv` có thêm `survival_time` (= real shipping days) + `event_delivered` -> CHỈ dùng cho **Cox PH / survival**, không dùng để train classification.
- Grain = order-item. Khi chia train/test phải **GROUP-AWARE theo `Order Id`** (các item cùng đơn chia sẻ một nhãn, không được tách 2 phía).