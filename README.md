# 🏠 HDB Resale Price Prediction — Full Competition Journey

> **Competition**: Regression Challenge (HDB Resale Price Prediction)  
> **Metric**: RMSE (Root Mean Squared Error) — lower is better  
> **Team**: TEAM 2  
> **Final Score**: 21,312.99869 (4th Place — Final Leaderboard)  
> **Best Personal Score**: 21,383 (blend_v23_v11_5_5_fixed)  
> **Best Team Score**: 21,312 (70% blend_V23_V11 + 30% meta-RF ensemble)  
> **Model**: CatBoost + LightGBM hybrid ensemble with native categorical encoding + meta-RF stacking

---

## 📋 Table of Contents

1. [What We Were Trying to Do](#what-we-were-trying-to-do)
2. [The Dataset](#the-dataset)
3. [Our Journey — Version by Version](#our-journey)
4. [Phase 10 — Block Composition & Feature Ceiling (V22–V29)](#phase-10--block-composition--feature-ceiling-v22v29)
5. [Phase 11 — Meta-RF Ensemble Breakthrough](#phase-11--meta-rf-ensemble-breakthrough)
6. [Key Turning Points](#key-turning-points)
7. [Feature Engineering Deep Dive](#feature-engineering-deep-dive)
8. [Final Feature List (V23 — Winning Base Model)](#final-feature-list-v23--winning-base-model)
9. [Domain Knowledge Ranked by Signal](#domain-knowledge-ranked-by-signal)
10. [Why CatBoost Beat Everything Else](#why-catboost-beat-everything-else)
11. [Challenges and How We Managed Them](#challenges-and-how-we-managed-them)
12. [Model Limitations](#model-limitations)
13. [Key Learnings](#key-learnings)
14. [What We Would Do With No Constraints](#what-we-would-do-with-no-constraints)
15. [Final Leaderboard History](#final-leaderboard-history)
16. [How to Reproduce](#how-to-reproduce)
17. [Academic References](#academic-references)

---

## 🎯 What We Were Trying to Do

Predict the **resale price of HDB flats in Singapore** as accurately as possible, given a transaction's physical attributes, location, and timing.

Think of it like this: a buyer looks at a flat and asks _"what is this worth?"_. Our model answers that question using 82 measurable signals — from floor area and storey level to which primary school is nearby and how the block was originally built.

**Why RMSE?** Root Mean Squared Error measures the average dollar gap between our predicted price and the actual sale price. An RMSE of 21,000 means our model is, on average, off by roughly $21,000 per transaction — about 4.7% of the typical $449,000 median price.

```
RMSE = sqrt( average of (predicted_price - actual_price)² )

Example:
  Actual:    $520,000
  Predicted: $498,000
  Error:     $22,000  → this gets squared, averaged, then square-rooted
```

---

## 📊 The Dataset

| Property      | Detail                               |
| ------------- | ------------------------------------ |
| Source        | Singapore HDB via Kaggle competition |
| Training rows | 150,634 transactions (2012–2023)     |
| Test rows     | 16,735 transactions                  |
| Features      | 77 columns per transaction           |
| Target        | `resale_price` (SGD)                 |
| Price range   | $150,000 – $1,258,000                |
| Median price  | $420,000                             |

### Feature Categories

The 77 columns covered six types of information:

```
┌─────────────────────────────────────────────────────────────┐
│  TRANSACTION    │ Year, month of sale                        │
│  PHYSICAL       │ Floor area, storey, flat type, age        │
│  LEASE          │ Remaining lease, commencement year         │
│  LOCATION       │ Town, block, street, planning area         │
│  TRANSPORT      │ MRT distance, bus stop, interchange status │
│  AMENITIES      │ Mall, hawker, school distances & counts    │
│  BLOCK UNITS    │ Unit type counts (1-room to exec, rental)  │
└─────────────────────────────────────────────────────────────┘
```

**Important note**: The competition prohibited external datasets — every feature had to come from the provided data files. This constraint shaped many of our decisions.

---

## 🗺️ Our Journey — Version by Version

### The Full Score Progression

```
         Phase 2          Phase 3   Phase 4   Phases 5-9       Phase 10     Phase 11
         LGB+XGB          CatBoost  Ensemble  CB+LGB Blends    Block Comp   Meta-RF
         ────────────     ────────  ────────  ─────────────    ──────────   ───────
22,500 ┤
       │  V1  V2  V3  V6
22,400 ┤  ●───●───●───●
       │               \
22,300 ┤                ●  V1 best: 22,309
22,000 ┤
21,900 ┤
       │                     V7 CatBoost ← BREAKTHROUGH (+694 pts)
21,800 ┤                     ●
21,700 ┤                      \ V8 seed ensemble
21,600 ┤                       \
       │                        \ V11 region features (+116 pts)
21,500 ┤                         ●
       │                          \ V12-V14 refinement
21,430 ┤                           ●─── V14+V11 blend ← Phase 9 best
       │
21,394 ┤                                V16+V11 blend ← Phase 10 start
       │
21,383 ┤                                 V23+V11 blend ← personal best
       │
21,312 ┤                                  70% V23+V11 + 30% meta-RF ← TEAM BEST
       │                              Group 9 ● 21,225
       └─────────────────────────────────────────────────────────────────▶

Total improvement: 22,309 → 21,312 = 997 points (↓4.5% RMSE)
```

---

### Phase 1 — Wrong Data Source (Before V1)

**What happened**: We initially downloaded raw CSV files from data.gov.sg, missing the rich location features the competition's official training set had.

**How we fixed it**: Switched to the official `train.csv` from Kaggle with 77 enriched features.

**Lesson**: Always verify your data source matches the competition's test set structure.

---

### Phase 2 — LightGBM + XGBoost Baseline (V1–V6)

Standard gradient boosting ensemble. Discovered data leakage from global mean encoding. Despite fixes, LGB/XGB stubbornly stayed around 22,300–22,459.

| Version | Features | Encoding        | LB Score           |
| ------- | -------- | --------------- | ------------------ |
| V1      | 43       | Global (leaked) | 22,309 ✅ best LGB |
| V2      | 43       | OOF (honest)    | 22,382             |
| V6      | 77       | 11 encodings    | 22,459 ❌ worst    |

---

### Phase 3 — CatBoost Breakthrough (V7–V8)

Switched to CatBoost with native categorical encoding. Immediate 694-point improvement.

```
V7: CatBoost, 45 features, native encoding → LB: 21,615  ← single biggest jump
V8: 3 seeds × 10 folds = 30 models         → LB: 21,538
```

**Why CatBoost won**: Handles raw categoricals internally via ordered target encoding. Eliminates the leakage problem that plagued LGB's OOF mean encoding.

---

### Phases 4–9 — CatBoost Refinement (V11–V14)

- V11: Singapore CCR/RCR/OCR region mapping → +116 pts
- V12: Primary/secondary school names as categoricals → +5 pts
- V13: StratifiedKFold on price deciles → +5 pts
- V14: CatBoost 90% + LightGBM 10% → 80-model ensemble
- **blend_v14_v11_5_5: V14 + V11 50/50 → LB: 21,430** ← Phase 9 best

---

## 🚀 Phase 10 — Block Composition & Feature Ceiling (V22–V29)

### V15 — MOP, Era Size, Cooling Measures

Three new feature families derived from Singapore HDB domain knowledge:

**MOP (Minimum Occupation Period) features:**

```python
mop_just_passed = 1 if hdb_age ∈ [5, 8]   # fresh MOP = motivated seller pool
is_mop_window   = 1 if hdb_age ∈ [5, 10]  # broader MOP market effect
years_past_mop  = (hdb_age - 5).clip(0)   # market stabilisation after MOP
```

**Era-relative flat size** — a 1990s 4-room flat (avg 100 sqm) vs a 2010s 4-room flat (avg 90 sqm) are different products despite the same flat type:

```python
area_vs_era     = floor_area - median(floor_area | flat_type, decade_built)
area_vs_era_pct = area_vs_era / cohort_median
```

**Cooling measure windows** — 5 ABSD/LTV tightening dates (Dec 2011, Jan 2013, Jun 2013, Jul 2018, Dec 2021):

```python
mths_2011_12 = months elapsed since Dec 2011 cooling (clipped to 0 before date)
months_since_last_cooling = min(all non-zero cooling month values)
```

**LB improvement**: ~21,404 → part of V15+V11 blend at 21,404

---

### V16 — Planning Area Maturity + School Demand

**Planning tier** (1–5 scale based on HDB official mature estate classification):

```python
PLANNING_TIER = {
    'CENTRAL AREA': 5, 'QUEENSTOWN': 5, 'BISHAN': 5,  # prime central
    'BUKIT MERAH': 4, 'SERANGOON': 4,                 # established mature
    'BEDOK': 3, 'TAMPINES': 3, 'SENGKANG': 3,         # mature outer
    'WOODLANDS': 2, 'YISHUN': 2, 'JURONG WEST': 2,    # non-mature
}
pt_x_flattype = planning_tier × flat_type_enc  # interaction term
```

r = +0.283 — confirmed by academic literature on mature estate premiums.

**School oversubscription** (MOE Phase 2C ballot data):

```python
OVERSUBSCRIPTION = {
    'Nanyang Primary School': 5,  # always ballots Phase 2C
    'Henry Park Primary School': 5,
    ...
}
oversub_in_1km    = oversubscription × (pri_sch_nearest_distance ≤ 1000m)
is_top_school_1km = 1 if oversub ≥ 4 AND dist ≤ 1km
```

**Why this works**: Singapore's P1 registration gives priority to children within 1km. Parents _specifically_ purchase flats to fall within the catchment — creating a documented, legally-consequential demand signal. Unlike secondary schools (no registration advantage by distance), primary schools create a genuine price cliff at 1km.

**LB: blend_v16_v11_5_5 → 21,394** (new team best at the time)

---

### V22 — Block Composition (Biggest New Signal Found)

Discovered that the dataset contained block-level unit type counts that had never been engineered into ratio features:

```python
total = df['total_dwelling_units'].clip(lower=1)

pct_premium  = (5room_sold + exec_sold) / total          # r = +0.521
pct_economy  = (1room_sold + 2room_sold + 3room_sold) / total  # r = −0.504
pct_rental   = total_rental_units / total                # r = −0.105
has_rental   = 1 if any rental units in block            # r = −0.178
has_commercial = 1 if commercial == 'Y'                  # r = −0.140

# Composite: full spectrum in one number
block_quality = pct_premium − pct_economy − pct_rental   # r = +0.578
```

**Why these are timeless signals**: Block composition is fixed at construction — how many 3-room vs 5-room flats a block has never changes. A premium block (high `pct_premium`) commands a consistent price premium regardless of the transaction year. These features are valid in 2012, 2021, and 2026.

**What was skipped (tested, confirmed weak)**:

- `pct_standard` (4-room only): r=+0.014 — 4-room is the benchmark, tells you nothing
- `pct_studio`: r=+0.026 — too rare (n=1,408)
- `year_completed`: r=0.999 corr with `lease_commence_date` — fully redundant

---

### V23 — Room Prestige Tier (Marginal)

Added 3-tier room prestige encoding to reduce MULTI-GENERATION (n=77) outlier noise:

```python
ROOM_PRESTIGE = {
    '5 ROOM': 3, 'EXECUTIVE': 3,   # premium tier
    '4 ROOM': 2,                    # standard tier
    '3 ROOM': 1, '2 ROOM': 1, '1 ROOM': 1, 'MULTI-GENERATION': 1,
}
```

**Result**: CB OOF improved +7 points (21,293). LGB regressed −133 points due to mean-encoding redundancy with existing `flat_type` categorical. Net blend OOF: +5 points.

**blend_v23_v11_5_5_fixed LB: 21,383 ← personal best**

---

### V24–V29 — The Feature Engineering Ceiling

Systematic pilot testing framework: Seed 42, 5 CB folds + 3 LGB folds (~45 min per pilot), kill/go decision vs V22 5-fold mean of 21,214.

| Version | Hypothesis                                          | CB 5-fold result                      | Decision          |
| ------- | --------------------------------------------------- | ------------------------------------- | ----------------- |
| V24     | Separate LGB feature matrix (no prestige cols)      | 21,461 all seeds                      | ❌ KILL           |
| V25     | Neighbourhood price memory (rolling 6m median)      | Leakage: 3,258 RMSE                   | ❌ KILL (leakage) |
| V26     | Town × MRT bin categorical + planning×log_mrt       | 21,517                                | ❌ KILL           |
| V27     | Bus×MRT interchange interaction + precinct pavilion | 21,517 (bus pct=23.3%, too broad)     | ❌ KILL           |
| V28     | K-Means spatial clusters (K=50, K=100) + cross-cat  | Fold 1: +576⚠️, systematic regression | ❌ KILL           |
| V29     | month_sin/cos + haversine CBD distance              | Running (CPU, no GPU)                 | Pending           |
| Huber   | Huber:delta=10,000 loss (wrong scale — log-space)   | 23,043 Fold 1                         | ❌ KILL           |

**Root cause diagnosis**: CatBoost at depth=9 with 50 models already extracts maximum signal from available features. Every addition since V22 was either:

1. Redundant with what CatBoost computes internally (room prestige → already in `flat_type` ordered encoding)
2. Creating fold-level noise from sparse joint categoricals (K-Means, town×MRT)
3. Applied at the wrong scale (Huber delta in dollar space vs log-transformed target)

**The information ceiling is real**: Top 1% of errors (1,507 rows) account for 18.7% of total MSE. These are premium flats where renovation quality, unit facing, and negotiation drive price — information not in the dataset. No regularisation or feature engineering can fix an information gap.

---

## 🤝 Phase 11 — Meta-RF Ensemble Breakthrough

### What the Teammate Built

A two-stage Random Forest meta-model with distinct feature engineering:

- K-Means spatial clusters (K=50, K=100) on (lat, lon, floor_area, mid_storey)
- `planning_area × flat_type` cross-category mean encoding
- Inverse distance features (1/mrt_dist, 1/mall_dist, 1/hawker_dist)
- Cyclical month encoding (sin/cos)

Meta-RF standalone OOF: **23,964** — terrible alone. But its errors were uncorrelated with CatBoost's errors on specific rows.

### The Blending Experiment

| V23+V11 weight | Meta-RF weight | LB Score              |
| -------------- | -------------- | --------------------- |
| 100%           | 0%             | 21,384                |
| 85%            | 15%            | 21,331                |
| **70%**        | **30%**        | **21,312 ✅ optimum** |
| 60%            | 40%            | 21,349                |

**The parabola bottomed at 70/30.** Each 15% shift from 100/0 toward 70/30 gained ~20 LB points. Going more aggressive past 70/30 started hurting.

### Why a Weak Model Helped

```
meta-RF OOF: 23,964  (standalone terrible)
CB+LGB OOF:  21,217  (standalone great)

But their error correlation is LOW on specific row segments.
Blending cancels errors on ~1,783 rows where one is right
and the other is wrong → net RMSE improvement of 71 points.

Key insight: An inaccurate but orthogonal model is more
valuable than a slightly better version of the same model.
```

### Final Team Score: 21,312 (4th Place)

In the final hour, two teams jumped from 2nd to 2nd/3rd with last-minute submissions. Final standings:

| #     | Team            | Score      |
| ----- | --------------- | ---------- |
| 1     | Group 9         | 21,225     |
| 2     | Team8_DS4       | 21,282     |
| 3     | DSAI4 Team 5    | 21,292     |
| **4** | **Team 2 (us)** | **21,313** |

Gap to 1st: 88 points. Likely explained by Group 9 having a clean implementation of neighbourhood price memory (rolling local market momentum) — the one signal we identified but couldn't implement without leakage.

---

## 💡 Key Turning Points

_(Preserving original Phase 2–9 turning points + new additions)_

### 1. CatBoost Replaced LGB/XGB (Phase 3)

Single biggest jump: 694 points. Native ordered encoding eliminated the leakage problem.

### 2. Block Composition Discovery (Phase 10, V22)

Block unit type counts (pct_premium, block_quality r=+0.578) were in the dataset throughout the competition but never engineered into ratio features until V22. Earlier discovery would have been worth 2–3 weeks of competition time.

### 3. Kill/Go Pilot Framework

Reducing from 10-fold to 5-fold pilots (Seed 42 only) saved ~80 hours of GPU/CPU time across 7 failed experiments. The framework: if CB Seed 42 OOF > threshold after 5 folds → kill. Never run a full version on a failing hypothesis.

### 4. Meta-RF Ensemble Insight (Phase 11)

A teammate's Random Forest with r=23,964 standalone gained 71 LB points at 30% weight. This taught the competition's most important lesson: ensemble diversity beats individual model quality.

### 5. Simple Blends Beat Complex Features

Adding school prestige features (hours of work) gained 5 LB points. Blending two existing submissions (minutes of work) gained 23 LB points. Always try the simple thing first.

---

## 🔬 Feature Engineering Deep Dive

_(Original sections preserved)_

### The Core Formula: Why log1p(price)?

```python
y = np.log1p(train['resale_price'])  # transform target
predictions = np.expm1(model.predict(X))  # inverse transform predictions
```

RMSE on log-prices penalises proportional errors equally — a $20K error on a $200K flat is treated the same as a $100K error on a $1M flat. This stabilises training and prevents the model from over-focusing on expensive flats.

---

## 📋 Final Feature List (V23 — Winning Base Model)

> V23 is the final base model used in `blend_v23_v11_5_5_fixed` (LB 21,383) and the 70/30 meta-RF blend (LB 21,312).
> V23 = V22 (80 features: V16 base + 6 block composition) + 3 room prestige features = **82 features total** (70 numeric + 12 categorical).

### Original Dataset Features (20)

| Feature                                  | Description                           |
| ---------------------------------------- | ------------------------------------- |
| `floor_area_sqm`                         | Floor area in square metres (r=+0.69) |
| `mid_storey`                             | Mid-point of storey range (r=+0.36)   |
| `max_floor_lvl`                          | Maximum floor level of block          |
| `hdb_age`                                | Age of flat at transaction            |
| `lease_commence_date`                    | Year lease started                    |
| `total_dwelling_units`                   | Total units in block                  |
| `Latitude`, `Longitude`                  | GPS coordinates                       |
| `mrt_nearest_distance`                   | Distance to nearest MRT (metres)      |
| `Mall_Nearest_Distance`                  | Distance to nearest mall              |
| `Hawker_Nearest_Distance`                | Distance to nearest hawker centre     |
| `pri_sch_nearest_distance`               | Distance to nearest primary school    |
| `Tranc_Year`, `Tranc_Month`              | Transaction date                      |
| `1room_sold` to `exec_sold`              | Block unit type counts (raw)          |
| `1room_rental` to `3room_rental`         | Rental unit counts (raw)              |
| `mrt_interchange`, `bus_interchange`     | Interchange station binary flags      |
| `Mall_Within_1km`, `Mall_Within_2km`     | Mall count thresholds                 |
| `Hawker_Within_1km`, `Hawker_Within_2km` | Hawker count thresholds               |
| `cutoff_point`, `affiliation`            | Secondary school quality signals      |

### Derived Features (62)

| Feature                               | Formula                                                      | Rationale                                                    |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `remaining_lease`                     | `99 - (Tranc_Year - lease_commence_date)`                    | CPF eligibility cliff at 70 years                            |
| `remaining_lease_sq`                  | `remaining_lease²`                                           | Non-linear lease value decay                                 |
| `floor_area_sq`                       | `floor_area_sqm²`                                            | Super-linear area premium                                    |
| `storey_ratio`                        | `mid_storey / (max_floor_lvl + 1)`                           | Relative height in building                                  |
| `time_index`                          | `(Tranc_Year-2012)×12 + Tranc_Month`                         | Linear market time                                           |
| `area_x_storey`                       | `floor_area × mid_storey`                                    | High floor + large flat interaction                          |
| `sqm_per_room`                        | `floor_area / n_rooms`                                       | Space quality per room                                       |
| `dist_cbd`                            | `√((lat-1.2837)²+(lon-103.8517)²)×111`                       | CBD proximity in km                                          |
| `log_mrt_dist`                        | `log1p(mrt_nearest_distance)`                                | Diminishing MRT proximity returns                            |
| `region`                              | District → CCR/RCR/OCR                                       | Singapore market zone                                        |
| `region_x_flattype`                   | `region + "_" + flat_type`                                   | Zone × room type                                             |
| `region_x_town`                       | `region + "_" + town`                                        | Zone × town                                                  |
| `planning_tier`                       | Town → 1–5 tier map                                          | Mature estate premium                                        |
| `is_mature_estate`                    | HDB official classification                                  | Binary mature flag                                           |
| `pt_x_flattype`                       | `planning_tier × flat_type_enc`                              | Tier × room interaction                                      |
| `oversubscription`                    | MOE Phase 2C ballot 0–5 score                                | School demand signal                                         |
| `oversub_in_1km`                      | `oversubscription × (pri_dist ≤ 1km)`                        | Legal P1 registration cliff                                  |
| `is_top_school_1km`                   | `1 if oversub ≥ 4 AND dist ≤ 1km`                            | Elite catchment binary                                       |
| `mop_just_passed`                     | `1 if hdb_age ∈ [5, 8]`                                      | Fresh MOP seller pool                                        |
| `is_mop_window`                       | `1 if hdb_age ∈ [5, 10]`                                     | Broader MOP effect                                           |
| `years_past_mop`                      | `(hdb_age - 5).clip(0)`                                      | Post-MOP stabilisation                                       |
| `area_vs_era`                         | `floor_area - cohort_median(flat_type, decade)`              | Large vs small for its era                                   |
| `area_vs_era_pct`                     | `area_vs_era / cohort_median`                                | Proportional era comparison                                  |
| `storey_vs_era`                       | `mid_storey - cohort_median(flat_type, decade)`              | Floor level vs era norm                                      |
| `mths_2011_12` to `mths_2021_12` (×5) | Months since each cooling date                               | Post-ABSD suppression                                        |
| `months_since_last_cooling`           | `min(non-zero cooling months)`                               | Distilled cooling signal                                     |
| `pct_premium`                         | `(5room_sold + exec_sold) / total_units`                     | Block design quality: r=+0.521                               |
| `pct_economy`                         | `(1+2+3room_sold) / total_units`                             | Economy concentration: r=−0.504                              |
| `pct_rental`                          | `total_rental / total_units`                                 | Social composition signal                                    |
| `has_rental`                          | `1 if any rental units`                                      | Binary stigma flag: r=−0.178                                 |
| `has_commercial`                      | `1 if commercial == 'Y'`                                     | Older block signal: r=−0.140                                 |
| `block_quality`                       | `pct_premium − pct_economy − pct_rental`                     | Composite: r=+0.578                                          |
| `amenity_score`                       | `1/(mrt/1000+0.1) + 1/(mall/1000+0.1) + 1/(school/1000+0.1)` | Combined convenience                                         |
| `decade_built`                        | `(lease_commence_date // 10) × 10`                           | Build era for cohort lookup                                  |
| `postal_district`                     | First 2 digits of postal code                                | District location                                            |
| `postal_sector`                       | First 3 digits of postal code                                | Sub-district location                                        |
| `room_prestige`                       | `MULTI-GEN/1-3 ROOM=1, 4 ROOM=2, 5 ROOM/EXEC=3`              | Tier signal cleaner than linear flat_type_enc — V23 addition |
| `is_premium_flat`                     | `1 if flat_type in {5 ROOM, EXECUTIVE}`                      | Binary premium flag — V23 addition                           |
| `premium_x_blockquality`              | `is_premium_flat × block_quality`                            | Premium flat in premium block joint signal — V23 addition    |

**Categoricals (12)**: `town`, `flat_type`, `flat_model`, `planning_area`, `mrt_name`, `full_flat_type`, `address`, `region`, `region_x_flattype`, `region_x_town`, `pri_sch_name`, `sec_sch_name`

---

## 🎯 Domain Knowledge Ranked by Signal

| Rank | Domain Signal               | Correlation | Source                                         |
| ---- | --------------------------- | ----------- | ---------------------------------------------- |
| 1    | **Block quality composite** | r=+0.578    | Block unit type counts — fixed at construction |
| 2    | **Flat type / room count**  | r=+0.70     | HDB flat classification                        |
| 3    | **Floor area**              | r=+0.69     | Physical measurement                           |
| 4    | **Premium unit ratio**      | r=+0.521    | 5-room + executive concentration per block     |
| 5    | **Economy unit ratio**      | r=−0.504    | 1-3 room concentration per block               |
| 6    | **Planning area maturity**  | r=+0.283    | HDB official mature estate classification      |
| 7    | **School oversubscription** | r=+0.186    | MOE Phase 2C ballot history                    |
| 8    | **Remaining lease**         | r=+0.35     | CPF withdrawal eligibility rules               |
| 9    | **Rental block stigma**     | r=−0.178    | Binary flag stronger than percentage           |
| 10   | **Commercial ground floor** | r=−0.140    | Older block design era                         |
| 11   | **MOP timing**              | qualitative | 5-year minimum occupation period cliff         |
| 12   | **Cooling measures**        | qualitative | 5 ABSD/LTV tightening dates 2011–2021          |
| 13   | **Era-relative flat size**  | qualitative | 1990s 4-room ≠ 2010s 4-room in sqm             |
| 14   | **CCR/RCR/OCR zones**       | qualitative | URA property market segmentation               |
| 15   | **Cultural floor numbers**  | qualitative | Floor 4=unlucky, floor 8=lucky (Chinese)       |

---

## 🧠 Why CatBoost Beat Everything Else

_(Original section preserved)_

CatBoost's **ordered target encoding** computes categorical statistics using only transactions that appear earlier in a randomly-ordered dataset — preventing the future-data leakage that plagued LightGBM's OOF mean encoding.

```
For each row, CatBoost computes:
  category_stat = mean(target for same category, in rows BEFORE current)

This is computed PER ROW, PER TREE, with a different random permutation.
Result: no row ever sees its own target value. Zero leakage by design.
```

For a dataset with 12 categorical features (town, flat_type, address, school names...), this means CatBoost can learn "Block 45 Stirling Road, 4-ROOM" as a specific market without ever peeking at that flat's own price.

---

## ⚠️ Challenges and How We Managed Them

_(Preserving original + new challenges)_

### Original Challenges (Phases 1–9)

**Challenge: Leakage in address encoding** — Global mean encoding uses all data including validation rows. Fixed with OOF encoding inside each fold.

**Challenge: V10 convergence failure** — LR=0.03 caused every fold to hit 25K iteration cap. Fixed by lowering LR to 0.02.

**Challenge: Submission row count mismatch** — test_preds npy array had 16,737 rows vs sample's 16,735. Fixed with `sample[['Id']].merge(sub_all, on='Id', how='left')` — never use positional slicing.

### New Challenges (Phases 10–11)

**Challenge: V25 Leakage in neighbourhood price memory**

Two bugs combined caused 3,258 RMSE (near-perfect = obvious leakage):

1. `hood_price_ratio = resale_price / hood_med_6m` — directly encodes the target as a feature
2. Global fallback median computed from full source_df including current rows

Fix: (1) Remove ratio feature entirely. (2) Fallback must use only prior-year transactions.

**Challenge: Huber loss at wrong scale**

`Huber:delta=10,000` designed for dollar-space errors applied to `log1p(price)` target where all residuals are 0.01–0.20 in magnitude. Result: Fold 1 RMSE 23,043 (+1,688 vs V22). Correct log-space deltas: 0.05–0.20.

**Challenge: K-Means cluster fold instability**

K=50 clusters with StratifiedKFold created sparse validation slices per cluster per fold. CatBoost's ordered encoding on thin cluster categories produced fold-dependent noise. Result: Fold 1 +576, Fold 2 +523, systematic regression across 4/5 folds.

**Challenge: V23 submission misalignment**

Raw test_preds `.npy` arrays had 16,737 rows (2 extra). Positional slicing to 16,735 rows gave wrong Id → prediction mapping. Fix: always build submission via `sample[['Id']].merge(sub_all, on='Id')`.

---

## 📉 Model Limitations

**What the model cannot predict well:**

1. **Premium flat outliers** (>$800K): Top 1% of errors account for 18.7% of total MSE. Renovation quality, specific unit facing, and buyer negotiation drive these prices — none is in the dataset.

2. **Market regime shifts**: StratifiedKFold creates artificially uniform validation distributions. The test set represents a specific time period, causing OOF to systematically outperform LB by ~166 points.

3. **Rare flat types**: MULTI-GENERATION (n=77), Studio Apartment (n=1,408) — too few examples for reliable prediction. These were explicitly excluded from derived features.

4. **Information ceiling**: RMSE of ~21,300 appears to be the hard floor for this feature set. The gap to Group 9 (~21,225) likely requires neighbourhood price memory (rolling local market momentum) — a temporally-safe target-encoded feature we identified but couldn't implement cleanly within the competition timeframe.

---

## 📚 Key Learnings

### From Phases 1–9 (Original)

1. CatBoost's ordered encoding solves the mean-encoding leakage problem by design
2. Verify convergence: `best_iter` must be well below `max_iterations`
3. Simple blends often beat complex features (V11+V12 blend +23 pts in minutes)
4. Model diversity beats model quality in ensembles (CB+LGB > CB+CB)
5. Always save OOF predictions to disk (`oof_predictions.npy`)

### From Phases 10–11 (New)

6. **Audit all dataset columns early**: Block composition features (r=+0.578 for `block_quality`) were in the dataset throughout but discovered only at V22. Earlier discovery would have been worth weeks of competition time.

7. **Pre-computing redundant features hurts**: CatBoost at depth=9 derives cross-feature interactions internally. Adding pre-computed versions of features it already has (room_prestige from flat_type, distance interactions from existing continuous features) creates noise, not signal.

8. **The feature engineering ceiling is real**: Seven consecutive pilot experiments (V23–V29) all failed to improve on V22. When every pilot regresses, accept the ceiling and pivot to ensemble diversity.

9. **A weak orthogonal model beats a strong correlated one**: meta-RF with OOF 23,964 gained 71 LB points at 30% weight. An inaccurate model with uncorrelated errors is more valuable than a slightly better version of the same model.

10. **Pilot framework saves GPU hours**: 5-fold Seed 42 pilots (~45 min) vs full 10-fold 5-seed runs (~8 hrs). Kill/go framework prevented wasting time on 7 losing experiments = ~56 GPU hours saved.

11. **Target-encoded temporal features require strict fold-level computation**: Any rolling statistic computed from `resale_price` must (a) use only prior-period data with strict temporal cutoff, (b) never include the current row's own transactions in any fallback, and (c) be recomputed inside each CV fold.

12. **Submission alignment**: Never use positional slicing on `.npy` arrays. Always `sample[['Id']].merge(sub_all, on='Id', how='left')`.

---

## 🚀 What We Would Do With No Constraints

_(Original section preserved)_

### 1. External Data Sources

| Data                               | Signal                                  | Expected Gain |
| ---------------------------------- | --------------------------------------- | ------------- |
| PropertyGuru listing prices        | Days on market, asking vs selling price | High          |
| URA private transaction data       | Private condo comparables nearby        | High          |
| HDB resale price index (quarterly) | Market cycle timing                     | Medium        |
| Interest rate history (MAS)        | Mortgage affordability shifts           | Medium        |
| Google Street View images          | Block condition, neighbourhood quality  | High          |
| School ballot results (annual)     | Actual oversubscription rate            | Medium        |

### 2. Neighbourhood Price Memory (Safe Implementation)

The single most impactful unmade feature. Rolling 6-month median of `(town × flat_type)` transactions using only data strictly prior to each row's transaction date. Key requirements for a leakage-safe implementation:

```python
# For each row with period P = Tranc_Year*12 + Tranc_Month:
# Use ONLY transactions where period < P (strict inequality)
# Minimum count threshold (≥5) before using the window
# Fallback hierarchy: (town×flat_type) → (town) → (flat_type global prior-year)
# Recompute INSIDE each CV fold using only that fold's training portion
# NEVER include resale_price derived ratios as features (target leakage)
```

Expected gain if implemented correctly: 80–150 OOF points.

### 3. Renovation and Condition Proxy

Scrape PropertyGuru listings for HDB flats, extract renovation keywords ("renovated 2022", "original condition") and build a renovation quality score. This alone could explain $50,000–$100,000 in price variance for the top 1% error rows.

### 4. Time-Aware Cross-Validation

Use time-based splits — train on 2012–2018, validate on 2019, etc. This prevents the OOF-to-LB gap caused by StratifiedKFold's artificial distribution uniformity and would give more accurate kill/go signals during experimentation.

### 5. Stacking with a Meta-Learner

```
Level 1: CatBoost OOF predictions (82 features)
Level 1: LightGBM OOF predictions  →  Ridge regression  →  Final prediction
Level 1: Random Forest OOF predictions
```

This learns optimal blend weights from data rather than fixing them manually — the principle our teammate's meta-RF demonstrated to work.

---

## 📈 Final Leaderboard History

| Version                 | Model                 | Key Change                     | LB Score   | Notes             |
| ----------------------- | --------------------- | ------------------------------ | ---------- | ----------------- |
| V1                      | LGB+XGB               | Baseline, global encoding      | 22,309     |                   |
| V7                      | CatBoost              | Native encoding — breakthrough | 21,615     | +694 pts          |
| V8                      | CatBoost              | 3-seed ensemble                | 21,538     |                   |
| V11                     | CatBoost              | Region CCR/RCR/OCR features    | 21,499     |                   |
| V12                     | CatBoost              | School name categoricals       | 21,494     |                   |
| blend_v11_v12           | CB+CB                 | 50/50 blend                    | 21,471     |                   |
| blend_v14_v11           | CB+LGB+CB             | Highest diversity              | 21,430     | Phase 9 best      |
| blend_v16_v11           | V16+V11               | Planning tier + school demand  | 21,394     |                   |
| **blend_v23_v11_fixed** | **V23+V11**           | **+ Block composition**        | **21,383** | **Personal best** |
| sub13 (85/15)           | V23+V11 + meta-RF     | First meta-RF attempt          | 21,331     | Teammate          |
| **70/30 meta-RF**       | **V23+V11 + meta-RF** | **Optimum blend**              | **21,312** | **Team best**     |
| sub21 (60/40)           | V23+V11 + meta-RF     | Too aggressive                 | 21,349     | Past optimum      |
| Various pilots          | V24–V29               | Feature experiments            | regressed  | All killed        |
| **FINAL**               | **Team 2**            | **4th place**                  | **21,313** |                   |

---

## 🔄 How to Reproduce

### Requirements

```bash
pip install catboost lightgbm scikit-learn pandas numpy
```

### Run the Winning Base Model (Kaggle)

1. Upload `train.csv`, `test.csv`, `sample_sub_reg.csv` to Kaggle dataset
2. Create new Kaggle notebook, set accelerator: **GPU T4**
3. Paste `kaggle_catboost_lgb_v23.py`, update `BASE_PATH`
4. **Save Version → Save & Run All (Commit)**
5. Download: `submission_kaggle_v23.csv`, all `oof_*.npy` arrays
6. Blend with V11: `(V23_predictions + V11_predictions) / 2`
7. Submit `blend_v23_v11_5_5_fixed.csv`

### Reproduce the Team Best (21,312)

1. Run step 1–5 above for V22/V23
2. Run teammate's meta-RF notebook (`HDB_Price_Regression_21312.ipynb`)
   - Uses K-Means clusters, planning_area×flat_type cross-encoding, inverse distances
3. Compute: `0.70 × blend_v23_v11 + 0.30 × meta_rf_predictions`
4. Align by Id: `sample[['Id']].merge(sub_all, on='Id', how='left')` — critical
5. Submit

### Expected Runtime

| Environment                  | V23 Full Run                           |
| ---------------------------- | -------------------------------------- |
| Kaggle GPU T4                | ~8–9 hours (80 models: 50 CB + 30 LGB) |
| Kaggle CPU (no GPU)          | ~35–40 hours                           |
| Pilot (5-fold, Seed 42 only) | ~45 min GPU / ~3–4 hrs CPU             |

---

## 📖 Academic References

1. Wang, L., Chan, F.F., Wang, Y., & Chang, Q. (2016). _Predicting Public Housing Prices Using Delayed Neural Networks_. IEEE TENCON 2016.
   - Key finding: Floor area, floor level, MRT distance, lease duration are top HDB price signals

2. Liu, Z., & Pan, Y. (2024). _Research of the Influence Factors of Housing Price — Take Singapore as an Example_. ICFTBA 2024.
   - Key finding: Flat model and lease commence date are collinear (r=0.999) — drop one
   - MLR R² = 0.674 with 4 clean features (floor area, flat type, storey, remaining lease)

3. Prokhorenkova, L. et al. (2018). _CatBoost: unbiased boosting with categorical features_. NeurIPS 2018.
   - The paper describing CatBoost's ordered encoding — the core of our winning approach

---

_Built for the Kaggle Regression Challenge (HDB Resale Price Prediction)_  
_Final Leaderboard: 4th Place — 21,312.99 RMSE_  
_Personal best: blend_v23_v11_5_5_fixed — 21,383 RMSE_  
_Team best: 70% blend_V23_V11 + 30% meta-RF — 21,312 RMSE_  
_Total journey: 22,309 → 21,312 = 997 points improvement (↓4.5% RMSE)_
