-- Supplier Lead-Time Risk Analysis
-- Run against SQLite DB built in src-code/02_sql_analysis.py

-- 1. Average lead time per supplier
SELECT
    supplier_id,
    COUNT(*)                          AS total_shipments,
    ROUND(AVG(lead_time_days), 2)     AS avg_lead_time,
    ROUND(MIN(lead_time_days), 2)     AS min_lead_time,
    ROUND(MAX(lead_time_days), 2)     AS max_lead_time
FROM shipment
JOIN customer USING (supplier_id)
GROUP BY supplier_id
ORDER BY avg_lead_time DESC;


-- 2. Delay frequency per supplier
SELECT
    supplier_id,
    COUNT(*)                                                        AS total_shipments,
    SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END)   AS delayed_count,
    ROUND(
        100.0 * SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) / COUNT(*),
        1
    )                                                               AS delay_rate_pct
FROM shipment
JOIN customer USING (supplier_id)
GROUP BY supplier_id
ORDER BY delay_rate_pct DESC;


-- 3. Volume-delay correlation (shipment volume vs delay rate by supplier)
SELECT
    supplier_id,
    COUNT(*)                                                        AS volume,
    ROUND(
        100.0 * SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) / COUNT(*),
        1
    )                                                               AS delay_rate_pct
FROM shipment
JOIN customer USING (supplier_id)
GROUP BY supplier_id
ORDER BY volume DESC;


-- 4. Monthly delay trends over time
SELECT
    strftime('%Y-%m', shipment_date)                                AS year_month,
    COUNT(*)                                                        AS total_shipments,
    SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END)   AS delayed_count,
    ROUND(
        100.0 * SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) / COUNT(*),
        1
    )                                                               AS delay_rate_pct
FROM shipment
GROUP BY year_month
ORDER BY year_month;


-- 5. Year-over-year delay growth rate
WITH yearly AS (
    SELECT
        strftime('%Y', shipment_date)                               AS yr,
        COUNT(*)                                                    AS total,
        SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) AS delayed
    FROM shipment
    GROUP BY yr
)
SELECT
    yr,
    total,
    delayed,
    ROUND(100.0 * delayed / total, 1)                               AS delay_rate_pct,
    ROUND(
        100.0 * (delayed - LAG(delayed) OVER (ORDER BY yr))
             / NULLIF(LAG(delayed) OVER (ORDER BY yr), 0),
        1
    )                                                               AS yoy_growth_pct
FROM yearly
ORDER BY yr;


-- 6. Penetration index (carrier share of delayed shipments vs overall share)
WITH totals AS (
    SELECT COUNT(*) AS grand_total,
           SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) AS total_delayed
    FROM shipment
),
carrier_stats AS (
    SELECT
        carrier,
        COUNT(*)                                                    AS carrier_volume,
        SUM(CASE WHEN delivery_status = 'Delayed' THEN 1 ELSE 0 END) AS carrier_delayed
    FROM shipment
    GROUP BY carrier
)
SELECT
    c.carrier,
    c.carrier_volume,
    c.carrier_delayed,
    ROUND(100.0 * c.carrier_volume    / t.grand_total,    1)       AS volume_share_pct,
    ROUND(100.0 * c.carrier_delayed   / t.total_delayed,  1)       AS delay_share_pct,
    ROUND(
        (1.0 * c.carrier_delayed / NULLIF(c.carrier_volume, 0))
      / (1.0 * t.total_delayed   / NULLIF(t.grand_total,   0)),
        2
    )                                                               AS penetration_index
FROM carrier_stats c, totals t
ORDER BY penetration_index DESC;
