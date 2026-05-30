-- Supplier Lead-Time & Late-Delivery Risk Analysis (DataCo Smart Supply Chain)
-- Run against the SQLite DB built in src-code/02_sql_analysis.py
-- (single table `deliveries`, loaded from Data/filtered/clean_data.csv).
--
-- The table is at order-ITEM grain. `Late_delivery_risk` is the binary target
-- (1 = late). Column names contain spaces/parentheses, so they are quoted with
-- double quotes per SQLite. Rates are computed at item grain, the table's
-- natural grain, so orders with more items contribute proportionally more rows.

-- 1. Late-delivery rate by shipping mode
SELECT
    "Shipping Mode"                                                AS shipping_mode,
    COUNT(*)                                                       AS total_items,
    SUM("Late_delivery_risk")                                      AS late_items,
    ROUND(100.0 * SUM("Late_delivery_risk") / COUNT(*), 1)         AS late_rate_pct,
    ROUND(AVG("Days for shipment (scheduled)"), 2)                 AS avg_scheduled_days
FROM deliveries
GROUP BY "Shipping Mode"
ORDER BY late_rate_pct DESC;


-- 2. Late-delivery rate by order region
SELECT
    "Order Region"                                                 AS order_region,
    COUNT(*)                                                       AS total_items,
    SUM("Late_delivery_risk")                                      AS late_items,
    ROUND(100.0 * SUM("Late_delivery_risk") / COUNT(*), 1)         AS late_rate_pct
FROM deliveries
GROUP BY "Order Region"
ORDER BY late_rate_pct DESC;


-- 3. Late-delivery rate by market
SELECT
    "Market"                                                       AS market,
    COUNT(*)                                                       AS total_items,
    SUM("Late_delivery_risk")                                      AS late_items,
    ROUND(100.0 * SUM("Late_delivery_risk") / COUNT(*), 1)         AS late_rate_pct,
    ROUND(AVG("Sales"), 2)                                         AS avg_sales
FROM deliveries
GROUP BY "Market"
ORDER BY late_rate_pct DESC;


-- 4. Late-delivery rate by product department
SELECT
    "Department Name"                                              AS department,
    COUNT(*)                                                       AS total_items,
    SUM("Late_delivery_risk")                                      AS late_items,
    ROUND(100.0 * SUM("Late_delivery_risk") / COUNT(*), 1)         AS late_rate_pct
FROM deliveries
GROUP BY "Department Name"
ORDER BY late_rate_pct DESC;


-- 5. Late-delivery rate by customer segment
SELECT
    "Customer Segment"                                             AS customer_segment,
    COUNT(*)                                                       AS total_items,
    SUM("Late_delivery_risk")                                      AS late_items,
    ROUND(100.0 * SUM("Late_delivery_risk") / COUNT(*), 1)         AS late_rate_pct,
    ROUND(AVG("Order Item Discount Rate"), 3)                      AS avg_discount_rate
FROM deliveries
GROUP BY "Customer Segment"
ORDER BY late_rate_pct DESC;


-- 6. Shipping-mode penetration index (mode's share of late deliveries vs share of volume)
WITH totals AS (
    SELECT COUNT(*)                  AS grand_total,
           SUM("Late_delivery_risk") AS total_late
    FROM deliveries
),
mode_stats AS (
    SELECT
        "Shipping Mode"           AS shipping_mode,
        COUNT(*)                  AS mode_volume,
        SUM("Late_delivery_risk") AS mode_late
    FROM deliveries
    GROUP BY "Shipping Mode"
)
SELECT
    m.shipping_mode,
    m.mode_volume,
    m.mode_late,
    ROUND(100.0 * m.mode_volume / t.grand_total, 1)                AS volume_share_pct,
    ROUND(100.0 * m.mode_late   / t.total_late,  1)                AS late_share_pct,
    ROUND(
        (1.0 * m.mode_late / NULLIF(m.mode_volume, 0))
      / (1.0 * t.total_late / NULLIF(t.grand_total, 0)),
        2
    )                                                              AS penetration_index
FROM mode_stats m, totals t
ORDER BY penetration_index DESC;
