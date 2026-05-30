# DataCo Dataset Assessment — Late-Delivery Risk

*Author: AI-assisted analysis (Claude). Date: 2026-05-30. Status: decision record for Group 8.*

This document records an **honest viability and modeling assessment** of the
DataCo Smart Supply Chain dataset for the (second) project pivot to
**late-delivery / supplier-risk prediction**. It is written to be cited in the
final report's *Data* and *Limitations* sections. All numbers below are
empirical, reproduced from `Data/filtered/clean_data.csv`.

## 1. Background — the second pivot

- **2026-05-29:** project pivoted *supplier lead-time → consumer credit risk*
  after the original supply-chain dataset's target proved to be synthetic
  leakage (ROC-AUC ≈ 0.52). Credit pivot reached XGBoost PR-AUC ≈ 0.91.
- **2026-05-30:** the supervisor said he had **confused the group's topic** and
  asked to return to supply chain using the **DataCo Smart Supply Chain**
  dataset (`shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`,
  CC0), retitled e.g. *"Supplier Lead-Time and Late Delivery Risk Prediction
  for Retail Procurement."* The group accepted.

## 2. Dataset at a glance

| Property | Value |
|---|---|
| Raw rows × cols | 180,519 × 53 (`DataCoSupplyChainDataset.csv`, latin-1) |
| Cleaned rows × cols | 180,519 × 29 (no impossible rows, 0 duplicates) |
| **Unique orders** | **65,752** (~2.75 line-items per order) |
| Target | `Late_delivery_risk` (binary, 1 = late) |
| Class balance | **54.8% late / 45.2% on-time** (near-balanced) |
| License | CC0 |
| Second file | `tokenized_access_logs.csv` — unrelated web clickstream, **unused** |

## 3. Leakage analysis (the central integrity step)

`Late_delivery_risk` is a **real observed outcome**, but several columns
*reconstruct* it and are dropped in `01_ingestion_cleaning.py`:

- `Late_delivery_risk` == (`Delivery Status` == "Late delivery") **exactly**.
- `Late_delivery_risk` == (`Days for shipping (real)` > `Days for shipment (scheduled)`) **~97.5%**.

**Dropped to prevent outcome leakage** (post-delivery, unknown at order time):
`Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`,
`Order Status`, `Benefit per order`, `Order Profit Per Order`,
`Order Item Profit Ratio`.
**Dropped as PII:** customer email, password, first/last name, street, zipcode.
**Kept** (decision-time inputs): scheduled days, shipping mode, market/region,
geography, product/category/department, customer segment, quantity, sales,
discount. This is the same leakage discipline used for the credit pivot.

## 4. Group structure — split must be order-aware

The data is at **order-item grain**. Within one `Order Id` the ~2.75 line-items
**share the same outcome** and share every signal-bearing feature (mode,
scheduled days, geography, segment); they differ only on product/qty/price,
which carry no lateness signal. Consequences:

- A plain random split leaks order-level patterns into the test set →
  **`GroupShuffleSplit` on `Order Id` is mandatory** everywhere.
- Item grain does **not** add information — it inflates N from 65,752 shipments
  to 180,519 rows. *"180,519 records" is technically true but the effective
  sample is ~65k shipments.* **Order grain is the cleaner modeling unit**
  (numbers barely move: 0.734 item-grain vs 0.752 order-grain ROC-AUC).

## 5. The decisive finding — signal collapses to one variable

`Shipping Mode` and `Days for shipment (scheduled)` are **perfectly collinear**:
each mode maps to exactly one scheduled-days value.

| Shipping Mode | Scheduled days | Late rate |
|---|---|---|
| Same Day | 0 | 45.7% |
| First Class | 1 | **95.3%** |
| Second Class | 2 | 76.6% |
| Standard Class | 4 | 38.1% |

Counter-intuitively, **faster shipping classes are late far more often** —
their promised windows are tight and routinely missed, while Standard Class has
4 days of slack. Late rates are essentially **flat (~55%) across every market,
region, customer segment, and department**, i.e. those carry almost no signal.

### Base-rate benchmark (run before reporting)

A trivial one-line rule — *predict each row's P(late) = the historical late rate
of its shipping mode* — was compared to the full model under the correct
group-aware split:

| Predictor | ROC-AUC | PR-AUC |
|---|---|---|
| One-line rule: P(late) = shipping-mode base rate | 0.725 | 0.749 |
| Lookup on mode **+** scheduled days | 0.725 (identical) | 0.749 |
| Full Random Forest (all features) | 0.734 | 0.812 |
| **Lift from machine learning** | **+0.009** | **+0.063** |

**Interpretation:** on ROC-AUC, four ML models add essentially nothing
(+0.01) over a one-line lookup; on PR-AUC they add a modest, real +0.06 (other
features help *rank within* a shipping mode). The flat segment rates and the
clean one-mode-one-SLA structure are the **fingerprint of a simulated dataset**
(DataCo is a teaching/simulated dataset — acceptable for coursework, but it must
be **disclosed**).

## 6. Honest verdict

**Usable and reportable — but a weaker analytical problem than the credit
dataset.** The signal is genuine and leak-free, but it concentrates in a single
variable (shipping mode). A third pivot would be the larger mistake; the right
move is to **report this honestly** rather than oversell a 0.73 AUC.

## 7. Reporting recommendations (turn the weakness into the contribution)

1. **Lead with the base-rate benchmark.** "Four models add +0.01 ROC-AUC over a
   one-line rule" is a sophisticated, honest result that earns marks.
2. **Reframe the threshold deliverable as business cost**, not class imbalance
   (the data is balanced, so `scale_pos_weight` / imbalance-weighting is now
   pointless): the asymmetric cost of flagging an on-time shipment vs. missing a
   late one.
3. **Add a residual-signal analysis:** *within* each shipping mode, does
   region / product / discount predict lateness? This recovers the multivariate
   story and is the genuinely open ML question.
4. **Cleanups:** model at **order grain**; drop one of the collinear pair
   `Sales per customer` / `Order Item Total`; disclose the simulated nature.

## 8. Decisions register

| # | Decision | Status |
|---|---|---|
| D1 | Adopt DataCo dataset; binary target `Late_delivery_risk` | ✅ accepted by group |
| D2 | Drop the leakage + PII columns listed in §3 | ✅ implemented in `01` |
| D3 | Group-aware split on `Order Id` everywhere | ✅ required; pending in `05` |
| D4 | Item grain vs order grain | ⬜ recommend order grain — **group to confirm** |
| D5 | Drop class-imbalance experiment; reframe threshold as business cost | ⬜ **group to confirm** |
| D6 | Disclose simulated dataset + base-rate benchmark in report | ⬜ **group to confirm** |

## 9. Migration status (code)

| Stage | State |
|---|---|
| `01_ingestion_cleaning.py` | ✅ migrated to DataCo |
| `02_sql_analysis.py` + `sql/analysis.sql` | ✅ migrated (6 delivery queries) |
| `03_eda.py` · `04_feature_engineering.py` · `05_modeling.py` · `06_visualization_advanced.py` | ⬜ still credit — pending |
| `app/app.py` · `app/data_quality_dashboard.py` | ⬜ still credit — pending |
| `report/main.tex` · CLAUDE.md downstream sections | ⬜ still credit — pending |
