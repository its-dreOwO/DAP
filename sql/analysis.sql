-- Consumer Credit-Default Risk Analysis
-- Run against the SQLite DB built in src-code/02_sql_analysis.py
-- (single table `credit`, loaded from Data/filtered/clean_data.csv).
--
-- The original supply-chain queries (monthly delay trends, year-over-year
-- growth, carrier penetration) do not apply: this dataset has no date column
-- and no carrier. They are replaced by the credit-risk analogues below.

-- 1. Default rate by loan grade
SELECT
    loan_grade,
    COUNT(*)                                                       AS total_loans,
    SUM(loan_status)                                               AS defaults,
    ROUND(100.0 * SUM(loan_status) / COUNT(*), 1)                  AS default_rate_pct,
    ROUND(AVG(loan_int_rate), 2)                                   AS avg_int_rate
FROM credit
GROUP BY loan_grade
ORDER BY loan_grade;


-- 2. Default rate by home ownership
SELECT
    person_home_ownership,
    COUNT(*)                                                       AS total_loans,
    SUM(loan_status)                                               AS defaults,
    ROUND(100.0 * SUM(loan_status) / COUNT(*), 1)                  AS default_rate_pct,
    ROUND(AVG(loan_amnt), 0)                                       AS avg_loan_amnt
FROM credit
GROUP BY person_home_ownership
ORDER BY default_rate_pct DESC;


-- 3. Default rate by loan intent
SELECT
    loan_intent,
    COUNT(*)                                                       AS total_loans,
    SUM(loan_status)                                               AS defaults,
    ROUND(100.0 * SUM(loan_status) / COUNT(*), 1)                  AS default_rate_pct,
    ROUND(AVG(loan_percent_income), 3)                            AS avg_loan_pct_income
FROM credit
GROUP BY loan_intent
ORDER BY default_rate_pct DESC;


-- 4. Default rate by income band
SELECT
    CASE
        WHEN person_income < 30000  THEN '1. <30k'
        WHEN person_income < 60000  THEN '2. 30k-60k'
        WHEN person_income < 100000 THEN '3. 60k-100k'
        ELSE                             '4. 100k+'
    END                                                            AS income_band,
    COUNT(*)                                                       AS total_loans,
    SUM(loan_status)                                               AS defaults,
    ROUND(100.0 * SUM(loan_status) / COUNT(*), 1)                  AS default_rate_pct
FROM credit
GROUP BY income_band
ORDER BY income_band;


-- 5. Default rate by loan-amount band
SELECT
    CASE
        WHEN loan_amnt < 5000   THEN '1. <5k'
        WHEN loan_amnt < 10000  THEN '2. 5k-10k'
        WHEN loan_amnt < 20000  THEN '3. 10k-20k'
        ELSE                         '4. 20k+'
    END                                                            AS loan_amount_band,
    COUNT(*)                                                       AS total_loans,
    SUM(loan_status)                                               AS defaults,
    ROUND(100.0 * SUM(loan_status) / COUNT(*), 1)                  AS default_rate_pct,
    ROUND(AVG(loan_percent_income), 3)                            AS avg_loan_pct_income
FROM credit
GROUP BY loan_amount_band
ORDER BY loan_amount_band;


-- 6. Grade penetration index (grade's share of defaults vs share of volume)
WITH totals AS (
    SELECT COUNT(*) AS grand_total,
           SUM(loan_status) AS total_defaults
    FROM credit
),
grade_stats AS (
    SELECT
        loan_grade,
        COUNT(*)         AS grade_volume,
        SUM(loan_status) AS grade_defaults
    FROM credit
    GROUP BY loan_grade
)
SELECT
    g.loan_grade,
    g.grade_volume,
    g.grade_defaults,
    ROUND(100.0 * g.grade_volume   / t.grand_total,    1)          AS volume_share_pct,
    ROUND(100.0 * g.grade_defaults / t.total_defaults, 1)          AS default_share_pct,
    ROUND(
        (1.0 * g.grade_defaults / NULLIF(g.grade_volume, 0))
      / (1.0 * t.total_defaults / NULLIF(t.grand_total,  0)),
        2
    )                                                              AS penetration_index
FROM grade_stats g, totals t
ORDER BY penetration_index DESC;
