"""
Clean DataCo — ALIGNED với contract nhóm (src-code/01_ingestion_cleaning.py)
Hướng: by Department (theo ý thầy) + giữ Survival analysis (đề bài gốc).

Nguyên tắc:
- Áp dụng ĐÚNG danh sách drop của contract: LEAKAGE / PII / EMPTY-REDUNDANT.
- Nhãn classification = Late_delivery_risk (cột gốc, theo contract).
- Giữ Order Id + Order Item Id để split GROUP-AWARE (grain = order-item).
- Lọc dòng lỗi (qty>0, sales>=0, scheduled>=0) + drop duplicates (theo contract).
- Department là feature/đơn vị phân tích chính (theo thầy).
- Survival analysis: tách ra FILE RIÊNG vì cần Days for shipping (real) làm
  duration — không trộn vào file classification để khỏi vi phạm anti-leakage.
"""
import pandas as pd
import numpy as np

SRC = "/home/user/workspace/uploaded_attachments/93b32b1721844bb6a236b36feab6f8c9/DataCoSupplyChainDataset.csv"
OUT_CLASS = "/home/user/workspace/clean_data.csv"            # khớp contract nhóm
OUT_SURV  = "/home/user/workspace/clean_data_survival.csv"   # cho Cox PH
OUT_REPORT = "/home/user/workspace/cleaning_report_aligned.md"

TARGET_COL = "Late_delivery_risk"
GROUP_KEY  = "Order Id"

# --- Danh sách drop COPY ĐÚNG từ contract nhóm ---
LEAKAGE_COLUMNS = [
    "Days for shipping (real)", "Delivery Status", "shipping date (DateOrders)",
    "Order Status", "Benefit per order", "Order Profit Per Order",
    "Order Item Profit Ratio",
]
PII_COLUMNS = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode",
]
EMPTY_OR_REDUNDANT_COLUMNS = [
    "Product Description", "Order Zipcode", "Product Image", "Product Status",
    "Category Id", "Department Id", "Product Card Id", "Product Category Id",
    "Order Item Cardprod Id", "Customer Id", "Order Customer Id",
]
NUMERIC_COLUMNS = [
    "Days for shipment (scheduled)", "Order Item Quantity", "Order Item Discount",
    "Order Item Discount Rate", "Order Item Product Price", "Product Price",
    "Sales", "Order Item Total", "Sales per customer",
]

log = []
raw = pd.read_csv(SRC, encoding="latin-1")
log.append(f"Input rows x cols: {raw.shape[0]:,} x {raw.shape[1]}")

# Giữ lại real days cho survival TRƯỚC khi drop
real_days = pd.to_numeric(raw["Days for shipping (real)"], errors="coerce")

cleaned = raw.copy()

# --- Drop leakage / PII / empty (chỉ cột có thật) ---
leak = [c for c in LEAKAGE_COLUMNS if c in cleaned.columns]
pii  = [c for c in PII_COLUMNS if c in cleaned.columns]
junk = [c for c in EMPTY_OR_REDUNDANT_COLUMNS if c in cleaned.columns]
cleaned = cleaned.drop(columns=leak + pii + junk)
log.append(f"Dropped LEAKAGE ({len(leak)}): {leak}")
log.append(f"Dropped PII ({len(pii)}): {pii}")
log.append(f"Dropped EMPTY/REDUNDANT ({len(junk)}): {junk}")

# --- Type-fix ---
for col in NUMERIC_COLUMNS:
    if col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
cleaned["order date (DateOrders)"] = pd.to_datetime(
    cleaned["order date (DateOrders)"], errors="coerce")

# --- Target binary {0,1} ---
cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce").astype(int)

# --- Lọc dòng lỗi (impossible) + duplicates (theo contract) ---
before = len(cleaned)
keep_mask = (cleaned["Order Item Quantity"] > 0) & (cleaned["Sales"] >= 0) & \
            (cleaned["Days for shipment (scheduled)"] >= 0)
cleaned = cleaned[keep_mask]
real_days = real_days[keep_mask.values]   # giữ đồng bộ cho survival
log.append(f"Impossible-value rows removed: {before - len(cleaned)}")

before = len(cleaned)
dup_mask = cleaned.duplicated()
cleaned = cleaned[~dup_mask].reset_index(drop=True)
real_days = real_days[~dup_mask.values].reset_index(drop=True)
log.append(f"Duplicate rows removed: {before - len(cleaned)}")

# --- Thêm time features từ order date (order-time, hợp lệ) ---
od = cleaned["order date (DateOrders)"]
cleaned["order_year"]    = od.dt.year
cleaned["order_month"]   = od.dt.month
cleaned["order_quarter"] = od.dt.quarter
cleaned["order_dow"]     = od.dt.dayofweek

log.append(f"Output (classification) rows x cols: {cleaned.shape[0]:,} x {cleaned.shape[1]}")
log.append(f"Unique orders (group key): {cleaned[GROUP_KEY].nunique():,}")
log.append(f"Items per order: {len(cleaned)/cleaned[GROUP_KEY].nunique():.2f}")
late_rate = cleaned[TARGET_COL].mean()*100
log.append(f"Target Late_delivery_risk -> late rate: {late_rate:.1f}% "
           f"(0={int((cleaned[TARGET_COL]==0).sum()):,}, 1={int((cleaned[TARGET_COL]==1).sum()):,})")

# --- Lưu file classification (khớp contract) ---
cleaned.to_csv(OUT_CLASS, index=False)
log.append(f"Saved classification dataset -> clean_data.csv")

# --- File SURVIVAL riêng: thêm duration (real days) + event ---
# Survival cần lead-time thực tế. Đơn 'Shipping canceled' đã không còn trong
# raw target? Vẫn có. survival_time = real days, event = 1 nếu giao (real>=0).
surv = cleaned.copy()
surv["survival_time"] = real_days.values        # duration tới khi giao
surv["event_delivered"] = (surv["survival_time"].notna() & (surv["survival_time"] >= 0)).astype(int)
surv = surv[surv["survival_time"].notna()]
surv.to_csv(OUT_SURV, index=False)
log.append(f"Saved survival dataset (with duration) -> clean_data_survival.csv "
           f"({surv.shape[0]:,} rows)")

# --- Bảng rủi ro theo Department (theo thầy) ---
dept = cleaned.groupby("Department Name").agg(
    n_orders=(TARGET_COL, "size"),
    late_rate=(TARGET_COL, "mean"),
    avg_scheduled=("Days for shipment (scheduled)", "mean"),
).reset_index()
dept["late_rate"] = (dept["late_rate"]*100).round(1)
dept["avg_scheduled"] = dept["avg_scheduled"].round(2)
dept = dept.sort_values("late_rate", ascending=False)

# --- Báo cáo ---
rep = ["# Báo cáo Cleaning — ALIGNED với contract nhóm\n",
       "Hướng: **by Department** (theo thầy) · Nhãn: **Late_delivery_risk** (theo contract nhóm) · Giữ survival riêng.\n",
       "## Các bước (khớp 01_ingestion_cleaning.py)\n"]
rep += [f"- {l}" for l in log]
rep.append("\n## Bảng rủi ro theo Department (late rate)\n")
rep.append("| Department | N orders | Late rate | Avg scheduled days |")
rep.append("|---|---|---|---|")
for _, r in dept.iterrows():
    rep.append(f"| {r['Department Name']} | {int(r['n_orders']):,} | {r['late_rate']}% | {r['avg_scheduled']} |")
rep.append("\n## Lưu ý quan trọng")
rep.append("- File `clean_data.csv` KHÔNG chứa cột leakage -> dùng cho **classification/modeling**.")
rep.append("- File `clean_data_survival.csv` có thêm `survival_time` (= real shipping days) "
           "+ `event_delivered` -> CHỈ dùng cho **Cox PH / survival**, không dùng để train classification.")
rep.append("- Grain = order-item. Khi chia train/test phải **GROUP-AWARE theo `Order Id`** "
           "(các item cùng đơn chia sẻ một nhãn, không được tách 2 phía).")

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(rep))

print("\n".join(log))
print("\nColumns (classification):", list(cleaned.columns))
