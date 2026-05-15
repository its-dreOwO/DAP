# Data Quality Log

## [2026-05-15] Missing `delivery_status` — Systematic Gap in shipment.csv

**Discovered by:** Nguyễn Hoài Khánh  
**File affected:** `Data/shipment.csv`  
**Total missing:** 24 rows (3.3% of 728)

### Pattern

All 24 missing rows share an identical profile:

| Field | Value |
|---|---|
| Type | Export |
| Product Category | Industrial Equipment |
| Origin Country | India |
| Destination Country | (missing — column blank in source) |
| Frequency | 1 shipment per month, every ~30 rows |
| Date range | 2024-01-10 → 2025-12-06 |

### Affected Shipment IDs

**2024:** SHP-2024-0009, 0039, 0069, 0099, 0129, 0159, 0189, 0219, 0249, 0279, 0309, 0339  
**2025:** SHP-2025-0009, 0039, 0069, 0099, 0129, 0159, 0189, 0219, 0249, 0279, 0309, 0339

### Assessment

This is **not random missingness** — the regularity (every 30th record, same category/origin, spanning 24 months) strongly suggests a **systematic tracking failure** for India → Industrial Equipment export routes, likely from a disconnected tracking system or an unreported route type.

### Action Taken

- 24 rows excluded from model training via `dropna(subset=["delivery_status"])` in `Data/filtered/extract_model_data.py`
- Final model dataset: **704 rows** (down from 728)
- These rows were **not imputed** — outcome is genuinely unknown

### Recommendation

Investigate whether a separate logistics tracking system handles India Industrial Equipment exports. If recoverable, these 24 shipments should be re-integrated before final model training.
