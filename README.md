# HDB Resale Price Prediction — Kaggle Competition

**Team 2** | Final LB Score: **21,312** (2nd Place) | Group 9 Target: 21,225

---

## Competition Overview

Predict Singapore HDB resale flat prices using machine learning. Training data spans 2012–2023 with 150,634 transactions. Test set: 16,735 rows. Metric: RMSE on actual dollar prices.

---

## Timeline of Activities

### Phase 1 — Baseline & Architecture (V11)
- Established CatBoost as primary model (3 seeds × 10 folds = 30 models)
- Introduced Singapore district → CCR/RCR/OCR region mapping from URA classification
- Key features: physical (floor area, storey, lease), location (MRT, mall, hawker, schools), postal sector
- **LB: ~21,600**

### Phase 2 — Feature Enrichment (V12–V15)
- Added non-linear transforms: `floor_area²`, `storey²`, `remaining_lease²`
- MOP (Minimum Occupation Period) cliff effect: `mop_just_passed`, `is_mop_window`, `years_past_mop`
- Era-relative flat size: how large/small a flat was *for its build decade*
- Government cooling measure timing: 5 policy dates, months-since-last-cooling
- Expanded to dual-model: CatBoost (50 models) + LightGBM (30 models)
- **LB: ~21,450**

### Phase 3 — Domain Signal Features (V16)
- Planning area maturity tiers (1–5 scale): CENTRAL AREA=5, OCR emerging=1
- School oversubscription (MOE Phase 2C ballot data, 0–5 score)
- `oversub_in_1km`: captures the legal 1km primary school registration cliff
- **Blend V16+V11 LB: 21,394 — new team best**

### Phase 4 — Block Composition (V22)
- Discovered high-signal unused columns: unit type counts per block
- `pct_premium` (5-room + executive ratio): r=+0.521
- `pct_economy` (1-3 room ratio): r=−0.504
- `block_quality` composite: r=+0.578 — strongest single new feature found
- `has_rental` binary flag: median −$113K premium flats near rental blocks
- **V22 CB OOF: ~21,300 — established as model ceiling**

### Phase 5 — Experimentation & Failures (V23–V29)
Multiple pilots testing: room prestige tiers, MRT×region interactions, K-Means spatial clusters, neighbourhood price memory, bus×MRT interchange, Huber loss. All regressed or showed marginal improvement. Root cause: CatBoost at depth=9 with 50 models already extracts maximum signal from available features.

**Key failure modes documented:**
- Leakage in neighbourhood price memory (V25) — temporal cutoff bug
- Redundancy: pre-computed features CatBoost already derives internally
- Fold variance from sparse joint categoricals (V28 K-Means)
- Wrong scale: Huber delta=10,000 applied to log-transformed target

### Phase 6 — Meta-RF Ensemble Breakthrough
- Teammate built a two-stage Random Forest meta-model with new features:
  - K-Means spatial clusters (K=50, K=100) on (lat, lon, area, floor)
  - `planning_area × flat_type` cross-category mean encoding
  - Inverse distance features (1/mrt_dist, 1/mall_dist)
- Meta-RF standalone OOF: 23,964 (weak alone)
- **70% blend_V23_V11 + 30% meta-RF → LB: 21,312 — FINAL BEST**

---

## Results Summary

| Version | Description | LB Score |
|---|---|---|
| V11 | CatBoost baseline + regions | ~21,600 |
| V15 | + MOP + era size + cooling | ~21,450 |
| V16+V11 blend | + Planning tier + school oversubscription | 21,394 |
| **V23+V11 blend** | **+ Block composition features** | **21,383** |
| **70% V23+V11 + 30% meta-RF** | **Teammate ensemble** | **21,312** |

---

## Features Used in Final Model (V22/V23 base)

### Original Dataset Features
| Feature | Type | Description |
|---|---|---|
| `floor_area_sqm` | Numeric | Floor area in square metres |
| `mid_storey` | Numeric | Mid-point of storey range |
| `max_floor_lvl` | Numeric | Maximum floor level of block |
| `hdb_age` | Numeric | Age of flat at time of transaction |
| `lease_commence_date` | Numeric | Year lease started |
| `total_dwelling_units` | Numeric | Total units in block |
| `Latitude`, `Longitude` | Numeric | GPS coordinates |
| `mrt_nearest_distance` | Numeric | Distance to nearest MRT (metres) |
| `Mall_Nearest_Distance` | Numeric | Distance to nearest mall (metres) |
| `Hawker_Nearest_Distance` | Numeric | Distance to nearest hawker centre |
| `pri_sch_nearest_distance` | Numeric | Distance to nearest primary school |
| `Tranc_Year`, `Tranc_Month` | Numeric | Transaction year and month |
| `town` | Categorical | HDB town (26 towns) |
| `flat_type` | Categorical | Room type (1 ROOM to MULTI-GEN) |
| `flat_model` | Categorical | Flat model design |
| `planning_area` | Categorical | URA planning area |
| `mrt_name` | Categorical | Nearest MRT station name |
| `pri_sch_name` | Categorical | Nearest primary school name |
| `1room_sold` to `exec_sold` | Numeric | Block unit type counts |
| `1room_rental` to `3room_rental` | Numeric | Rental unit counts |

### Derived Features
| Feature | Formula | Rationale |
|---|---|---|
| `remaining_lease` | `99 - (Tranc_Year - lease_commence_date)` | CPF eligibility and buyer willingness to pay declines with lease |
| `remaining_lease_sq` | `remaining_lease²` | Non-linear lease cliff below 70 years |
| `floor_area_sq` | `floor_area_sqm²` | Large flats command super-linear premium |
| `storey_ratio` | `mid_storey / (max_floor_lvl + 1)` | Relative height in building — independent of absolute level |
| `time_index` | `(Tranc_Year - 2012) × 12 + Tranc_Month` | Linear market time encoding |
| `area_x_storey` | `floor_area × mid_storey` | High floor + large flat = non-additive premium |
| `sqm_per_room` | `floor_area / n_rooms` | Space quality per room |
| `dist_cbd` | `√((lat-1.2837)² + (lon-103.8517)²) × 111` | CBD proximity premium in km |
| `log_mrt_dist` | `log1p(mrt_nearest_distance)` | Diminishing returns of proximity |
| `region` | District → CCR/RCR/OCR map | Singapore property market zones |
| `region_x_flattype` | `region + "_" + flat_type` | Zone × room type interaction |
| `region_x_town` | `region + "_" + town` | Zone × town interaction |
| `planning_tier` | Town → 1–5 tier map | Mature estate price premium |
| `is_mature_estate` | Binary HDB classification | Official mature vs non-mature |
| `pt_x_flattype` | `planning_tier × flat_type_enc` | Tier 5 executive ≠ Tier 2 executive |
| `oversubscription` | MOE P1 ballot score 0–5 | School demand proxy (legally consequential 1km threshold) |
| `oversub_in_1km` | `oversubscription × (pri_dist ≤ 1000m)` | Captures P1 registration cliff effect |
| `is_top_school_1km` | `1 if oversub ≥ 4 and dist ≤ 1km` | Binary elite school catchment flag |
| `mop_just_passed` | `1 if hdb_age ∈ [5, 8]` | Fresh MOP = motivated seller pool |
| `is_mop_window` | `1 if hdb_age ∈ [5, 10]` | Broader MOP market effect |
| `years_past_mop` | `(hdb_age - 5).clip(0)` | Market stabilisation after MOP |
| `area_vs_era` | `floor_area - cohort_median(flat_type, decade)` | Was this flat large for its era? |
| `area_vs_era_pct` | `area_vs_era / cohort_median` | Proportional era comparison |
| `mths_{year}_{month}` (×5) | Months since each cooling measure date | Post-ABSD market suppression duration |
| `months_since_last_cooling` | `min(non-zero cooling months)` | Single distilled cooling signal |
| `pct_premium` | `(5room_sold + exec_sold) / total_units` | Block design era quality: r=+0.521 |
| `pct_economy` | `(1+2+3room_sold) / total_units` | Economy flat concentration: r=−0.504 |
| `pct_rental` | `total_rental / total_units` | Social composition signal |
| `has_rental` | `1 if any rental units in block` | Binary stigma: median −$113K effect |
| `has_commercial` | `1 if commercial == 'Y'` | Older block design proxy |
| `block_quality` | `pct_premium − pct_economy − pct_rental` | Composite block quality: r=+0.578 |
| `amenity_score` | `1/(mrt/1000+0.1) + 1/(mall/1000+0.1) + 1/(school/1000+0.1)` | Combined convenience index |

---

## Domain Knowledge Used (Ranked by Signal)

| Rank | Domain Signal | Correlation | Notes |
|---|---|---|---|
| 1 | Block composition quality (`block_quality`) | r=+0.578 | HDB blocks fixed at construction — timeless signal |
| 2 | Floor area (`floor_area_sqm`) | r=+0.69 | Largest single predictor |
| 3 | Flat type / room count | r=+0.70 | via categorical encoding |
| 4 | Premium unit ratio (`pct_premium`) | r=+0.521 | 5-room + executive concentration |
| 5 | Economy unit ratio (`pct_economy`) | r=−0.504 | 1-3 room concentration |
| 6 | Planning area maturity tier | r=+0.283 | HDB mature estate classification |
| 7 | School oversubscription (1km) | r=+0.186 | MOE P1 Phase 2C ballot data |
| 8 | Remaining lease | r=+0.35 | CPF withdrawal rules drive cliff |
| 9 | Rental block stigma (`has_rental`) | r=−0.178 | Binary flag stronger than percentage |
| 10 | MOP timing | qualitative | 5-year minimum occupation period cliff |
| 11 | Cooling measures (5 dates) | qualitative | ABSD/LTV tightening policy events |
| 12 | Era-relative flat size | qualitative | 1990s 4-room vs 2010s 4-room differ significantly |
| 13 | Cultural floor numbers | qualitative | Floor 4 = unlucky (4=death), Floor 8 = lucky in Chinese culture |
| 14 | CCR/RCR/OCR zones | qualitative | URA market segmentation |
| 15 | Commercial ground floor (`has_commercial`) | r=−0.140 | Older block design proxy |

---

## Challenges & Learnings

**Challenge 1 — Feature ceiling reached early.** V22 established a hard ceiling. Every subsequent feature addition regressed or was marginal. CatBoost at depth=9 with 50 models already extracts cross-feature interactions from available data. Pre-computing interactions that the model derives internally just adds noise.

**Challenge 2 — Leakage in neighbourhood price memory.** V25 attempted rolling median (town × flat_type) by prior months. A two-bug combination — `hood_price_ratio` encoding the target directly, and the global fallback using full-source data — caused 3,258 RMSE (near-perfect = obvious leakage). Temporal cutoff logic for target-encoded features requires strict fold-level computation.

**Challenge 3 — OOF optimism from StratifiedKFold.** Stratifying on price deciles creates artificially uniform validation distributions. The test set reflects a specific time period (not a stratified blend of all years), so OOF systematically outperformed LB by ~166 points. Fold 3 was always the best fold because its validation set concentrated stable-market-period transactions.

**Challenge 4 — Huber delta at wrong scale.** Huber:delta=10,000 was designed for dollar-space errors, but the model trains on log1p(price) where all residuals are between 0.01–0.20. The delta should have been 0.05–0.20 in log-space. Result: 23,043 RMSE on Fold 1 vs V22's 21,355.

**Learning 1 — Diversity beats accuracy in ensembles.** The meta-RF has OOF of 23,964 alone — terrible. But its errors are uncorrelated with CatBoost's errors on specific rows. 30% weight improved LB by 71 points. An inaccurate but orthogonal model is more valuable than a slightly better version of the same model.

**Learning 2 — Audit all dataset columns before engineering.** Block composition features (pct_premium, block_quality) were in the dataset throughout the competition but discovered only at V22. Earlier discovery would have closed the gap with Group 9 sooner.

**Learning 3 — Pilot before committing.** Reducing from 10-fold to 5-fold (then 3-fold) pilots saved ~80 hours of GPU/CPU time. The kill/go framework (kill if CB Seed 42 OOF > threshold) prevented wasted full runs on losing experiments.

**Learning 4 — The information ceiling is real.** Top 1% of errors (1,507 rows) account for 18.7% of total MSE. These are premium flats where renovation quality, unit facing, and negotiation — all invisible to the model — drive the price. No regularisation technique (DART, Huber, winsorisation) can fix an information gap.
