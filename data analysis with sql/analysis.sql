-- ============================================================
-- BƯỚC 3: DATA ANALYSIS WITH SQL (SQLite)
-- Dữ liệu: clean_data.csv (giữ NGUYÊN tên cột gốc, có dấu cách)
-- Bảng: orders
--
-- ⚠️ QUAN TRỌNG: tên cột gốc có DẤU CÁCH và VIẾT HOA, ví dụ:
--    "Department Name", "Shipping Mode", "Order Id", "Customer Segment"
-- Trong SQLite, cột có dấu cách PHẢI bọc trong dấu ngoặc kép "..."
-- Nếu không sẽ báo lỗi: no such column: department_name
--
-- Nhãn dùng: Late_delivery_risk (1 = trễ, 0 = đúng giờ/sớm)
--   -> AVG("Late_delivery_risk") = tỷ lệ trễ. Nhân 100 ra phần trăm.
-- ============================================================


-- ============================================================
-- Q1. TỔNG QUAN: bao nhiêu đơn, bao nhiêu đơn trễ?
-- ============================================================
SELECT
    COUNT(*)                                  AS total_rows,
    SUM("Late_delivery_risk")                 AS late_orders,
    COUNT(*) - SUM("Late_delivery_risk")      AS on_time_orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)   AS late_rate_pct
FROM orders;


-- ============================================================
-- Q2. GRAIN: dữ liệu ở mức order-item, mỗi đơn nhiều dòng
-- ============================================================
SELECT
    COUNT(*)                      AS order_item_rows,
    COUNT(DISTINCT "Order Id")    AS unique_orders,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT "Order Id"), 2) AS items_per_order
FROM orders;


-- ============================================================
-- Q3. TỶ LỆ TRỄ THEO DEPARTMENT  (yêu cầu đề bài)
-- ============================================================
SELECT
    "Department Name"                         AS department,
    COUNT(*)                                  AS orders,
    SUM("Late_delivery_risk")                 AS late_orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)   AS late_rate_pct
FROM orders
GROUP BY "Department Name"
ORDER BY late_rate_pct DESC;


-- ============================================================
-- Q4. THỜI GIAN GIAO DỰ KIẾN TRUNG BÌNH THEO DEPARTMENT
-- ============================================================
SELECT
    "Department Name"                                 AS department,
    COUNT(*)                                          AS orders,
    ROUND(AVG("Days for shipment (scheduled)"), 2)    AS avg_scheduled_days
FROM orders
GROUP BY "Department Name"
ORDER BY avg_scheduled_days DESC;


-- ============================================================
-- Q5. TỶ LỆ TRỄ THEO SHIPPING MODE  ⭐ yếu tố mạnh nhất
-- ============================================================
SELECT
    "Shipping Mode"                                   AS shipping_mode,
    COUNT(*)                                          AS orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)           AS late_rate_pct,
    ROUND(AVG("Days for shipment (scheduled)"), 2)    AS avg_scheduled_days
FROM orders
GROUP BY "Shipping Mode"
ORDER BY late_rate_pct DESC;


-- ============================================================
-- Q6. TỶ LỆ TRỄ THEO MARKET (thị trường lớn)
-- (Market không có dấu cách nên không bắt buộc ngoặc kép, nhưng để cho đồng bộ)
-- ============================================================
SELECT
    "Market"                                  AS market,
    COUNT(*)                                  AS orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)   AS late_rate_pct
FROM orders
GROUP BY "Market"
ORDER BY late_rate_pct DESC;


-- ============================================================
-- Q7. XU HƯỚNG TRỄ THEO NĂM (trend over time)
-- ============================================================
SELECT
    order_year                                AS year,
    COUNT(*)                                  AS orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)   AS late_rate_pct
FROM orders
GROUP BY order_year
ORDER BY year;


-- ============================================================
-- Q8. BẢNG RISK SCORECARD THEO DEPARTMENT  (dùng cho Power BI)
-- ============================================================
SELECT
    "Department Name"                                 AS department,
    COUNT(*)                                          AS orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)           AS late_rate_pct,
    ROUND(AVG("Days for shipment (scheduled)"), 2)    AS avg_scheduled_days,
    ROUND(AVG("Sales"), 2)                            AS avg_sales
FROM orders
GROUP BY "Department Name"
ORDER BY late_rate_pct DESC;


-- ============================================================
-- Q9. TOP TỔ HỢP DEPARTMENT × SHIPPING MODE RỦI RO NHẤT
-- ============================================================
SELECT
    "Department Name"                         AS department,
    "Shipping Mode"                           AS shipping_mode,
    COUNT(*)                                  AS orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)   AS late_rate_pct
FROM orders
GROUP BY "Department Name", "Shipping Mode"
HAVING COUNT(*) >= 100
ORDER BY late_rate_pct DESC
LIMIT 10;


-- ============================================================
-- Q10. TỶ LỆ TRỄ THEO PHÂN KHÚC KHÁCH HÀNG (customer segment)
-- ============================================================
SELECT
    "Customer Segment"                        AS customer_segment,
    COUNT(*)                                  AS orders,
    ROUND(AVG("Late_delivery_risk")*100, 1)   AS late_rate_pct
FROM orders
GROUP BY "Customer Segment"
ORDER BY late_rate_pct DESC;