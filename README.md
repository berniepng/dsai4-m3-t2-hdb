# 🏆 HDB Resale Price Prediction — Kaggle #1 Solution

> **Final Leaderboard Score: 21,615.51 (RMSE) — 1st Place**  
> Competition: Regression Challenge (HDB Resale Price Prediction)  
> Model: CatBoost with native categorical encoding, 5-fold cross-validation

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [The Full Journey — What We Tried](#the-full-journey)
5. [Feature Engineering](#feature-engineering)
6. [Feature Selection Process](#feature-selection-process)
7. [Model Evolution](#model-evolution)
8. [Why CatBoost Won](#why-catboost-won)
9. [Hyperparameters](#hyperparameters)
10. [Key Lessons Learned](#key-lessons-learned)
11. [How to Reproduce](#how-to-reproduce)

---

## 🏠 Project Overview

This repository documents the end-to-end solution for predicting Singapore HDB (Housing Development Board) resale flat prices — from raw data exploration through feature engineering, multiple model iterations, and the final winning submission.

**The core finding: CatBoost's native ordered target encoding fundamentally outperforms manually-engineered mean encodings for this dataset**, eliminating target leakage while preserving the full signal from high-cardinality categorical features like address, MRT station name, and flat type.

### Results Summary

| Submission | Model | LB RMSE | Notes |
|---|---|---|---|
| submission_official | LGB + XGB | 22,309 | Baseline — global mean encoding |
| submission_oof | LGB + XGB | 22,382 | OOF encoding — more honest |
| submission_v3 | LGB + XGB | 22,394 | 60 features, OOF |
| submission_kaggle_v4 | LGB + XGB | 22,378 | 70 features, OOF, 20K trees |
| submission_kaggle_v5 | LGB + XGB | 22,379 | Global encoding, 70 features |
| submission_kaggle_v6 | LGB + XGB | 22,459 | 77 features, 11 encodings |
| **submission_kaggle_v7** | **CatBoost** | **21,615** | **🏆 #1 — 45 features, native encoding** |

---

## 📊 Dataset

- **Source**: Singapore Housing Development Board (HDB) via Kaggle competition
- **Training set**: 150,634 transactions (2012–2021)
- **Test set**: 16,735 transactions
- **Target**: `resale_price` (SGD)
- **Features**: 77 columns including property attributes, location distances, school data, and amenity information
- **No external datasets used** — competition rules restricted all features to provided data only

### Key Statistics
```
Train resale_price: mean=$449,162  median=$420,000  std=$143,308
Price range: $150,000 – $1,258,000
Transaction years: 2012–2021
Towns: 26 unique HDB towns
Flat types: 7 (1 ROOM to MULTI-GENERATION)
```

---

## 📁 Repository Structure

```
hdb-price-prediction/
│
├── README.md                          # This file
│
├── data/
│   └── README.md                      # Data description (files not included — Kaggle source)
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb   # Feature development and correlation analysis
│   └── 03_model_experiments.ipynb     # All model iterations and experiments
│
├── src/
│   ├── features.py                    # Feature engineering functions
│   ├── models.py                      # Model training utilities
│   └── utils.py                       # Helper functions
│
├── submissions/
│   ├── submission_official.csv        # V1 — LGB+XGB baseline
│   ├── submission_oof.csv             # V2 — OOF encoding
│   ├── submission_kaggle_v3.csv       # V3 — 60 features
│   ├── submission_kaggle_v4.csv       # V4 — 70 features, 20K trees
│   ├── submission_kaggle_v5.csv       # V5 — global encoding
│   ├── submission_kaggle_v6.csv       # V6 — 77 features, 11 encodings
│   └── submission_kaggle_v7.csv       # V7 — CatBoost WINNER 🏆
│
├── results/
│   ├── leaderboard_progression.png    # Score progression chart
│   └── feature_importance.png         # CatBoost feature importance
│
├── docs/
│   ├── whitepaper_ntu.pdf             # NTU paper on HDB price prediction
│   └── whitepaper_liu_pan.pdf         # Liu & Pan 2024 factor analysis
│
├── kaggle_catboost_v7.py              # 🏆 WINNING SCRIPT — run this on Kaggle
├── kaggle_notebook_v6.py              # Previous best LGB+XGB attempt
└── requirements.txt                   # Python dependencies
```

---

## 🗺️ The Full Journey

### Phase 1 — Data Source Investigation

The competition provided 5 raw CSV files from [data.gov.sg](https://data.gov.sg/collections/189/view) covering HDB resale transactions from 1990–2026. The test set was an **enriched version** with 77 features including MRT proximity, school distances, mall access, and geo-coordinates — features not present in the raw gov data.

**Key discovery**: The `test.csv` provided by Kaggle contained the same 76 features as the training set. The initial approach of using the raw gov.sg CSVs was therefore **using the wrong training data** — the official `train.csv` from the competition had the full enriched feature set.

### Phase 2 — Academic Foundation

Two white papers informed feature selection:

**NTU Paper (Wang et al., 2016)** — identified 6 intrinsic HDB value drivers:
- Number of rooms, floor level, floor area, lease duration, distance to school, distance to MRT

**Liu & Pan (ICFTBA 2024)** — confirmed via MLR that:
- Floor area (r=0.69) and flat type (r=0.70) are top signals
- Remaining lease is significant; flat model and lease commence date are collinear and droppable in linear models
- VIF analysis: remaining lease and lease commence date r=0.999 → drop one for linear models

### Phase 3 — Baseline Model

**submission_official (LB: 22,309)**  
- LightGBM + XGBoost, 43 features  
- Global `addr_mean_price` encoding  
- Simple 50/50 blend  
- ~516 LGB trees, ~1,182 XGB trees at LR=0.05  

This became the benchmark to beat. Every subsequent attempt for 6 versions failed to improve on it.

### Phase 4 — The Leakage Problem

The critical issue with all LGB/XGB versions was **target encoding leakage**:

```
Global mean encoding:
  addr_mean_price = avg(resale_price) for each address
                    ← computed from ALL 150,634 rows
                    ← validation rows' prices are included
                    ← OOF RMSE is artificially inflated
```

This created a systematic ~750 point gap between OOF RMSE and leaderboard RMSE across all versions. The model "learned" to trust `addr_mean_price` as a near-perfect predictor — but on truly unseen test data, this signal was weaker.

| Version | OOF RMSE | LB RMSE | Gap (leakage) |
|---|---|---|---|
| V5 (global) | 21,610 | 22,379 | **769 points** |
| V6 (global+11 enc) | 21,669 | 22,459 | **790 points** |
| V7 CatBoost | 21,769 | 21,615 | **−154 points** ✅ |

V7 actually scored **better** on the leaderboard than OOF — because CatBoost's ordered encoding is properly conservative, sometimes slightly *underestimating* training performance.

---

## ⚙️ Feature Engineering

### Features KEPT (final 45 for V7)

**Temporal (3)**
| Feature | Description | Correlation |
|---|---|---|
| `Tranc_Year` | Transaction year | Market cycle signal |
| `Tranc_Month` | Transaction month | Seasonal effects |
| `time_index` | Months since Jan 2012 | Continuous time signal |

**Physical Property (8)**
| Feature | Description | Correlation |
|---|---|---|
| `floor_area_sqm` | Flat size in sqm | +0.654 |
| `mid_storey` | Floor midpoint | +0.353 |
| `max_floor_lvl` | Block height | +0.496 |
| `hdb_age` | Age at transaction | −0.350 |
| `lease_commence_date` | Build year | +0.350 |
| `remaining_lease` | Years of lease left | +0.362 |
| `total_dwelling_units` | Units in block | −0.141 |
| `flat_type_enc` | Ordinal room encoding | +0.663 |

**Engineered Interactions (4)**
| Feature | Formula | Rationale |
|---|---|---|
| `area_x_storey` | floor_area × mid_storey | Size-height premium |
| `area_per_storey` | floor_area / (mid_storey+1) | Normalised area |
| `storey_ratio` | mid_storey / max_floor_lvl | Relative floor position |
| `lease_x_area` | remaining_lease × floor_area | Lease-size composite |

**Distance Features (12)**
- MRT: `mrt_nearest_distance`, `log_mrt_dist`, `mrt_interchange`, `bus_interchange`
- Mall: `Mall_Nearest_Distance`, `log_mall_dist`, `Mall_Within_1km`, `Mall_Within_2km`
- Hawker: `Hawker_Nearest_Distance`, `log_hawker_dist`, `Hawker_Within_1km`, `Hawker_Within_2km`

**School & Amenity (6)**
- `pri_sch_nearest_distance`, `pri_sch_affiliation`
- `sec_sch_nearest_dist`, `cutoff_point`, `affiliation`
- `bus_stop_nearest`, `hawker_food_stalls`

**Geo (4)**
- `Latitude`, `Longitude`, `dist_cbd`, `multistorey_carpark`

**Categorical — Native CatBoost (7)**
| Feature | Unique Values | Why Included |
|---|---|---|
| `town` | 26 | Primary location signal |
| `flat_type` | 7 | Room count category |
| `flat_model` | 20 | Build quality/era |
| `planning_area` | 32 | Sub-region signal |
| `mrt_name` | 94 | Nearest MRT station |
| `full_flat_type` | 43 | Most granular flat descriptor |
| `address` | 9,157 | Block-level location |

---

## 🔬 Feature Selection Process

### Step 1 — Correlation Screening

Initial Pearson correlation with `resale_price`:

```
STRONG (|r| > 0.4):   flat_type_enc, floor_area_sqm, max_floor_lvl
MEDIUM (|r| > 0.2):   mid_storey, remaining_lease, hdb_age, dist_cbd
WEAK   (|r| > 0.1):   mrt_nearest_distance, total_dwelling_units
NOISE  (|r| < 0.1):   month_num, hawker_market_stalls, vacancy
```

### Step 2 — Multicollinearity Check

Following Liu & Pan (2024) VIF analysis:
- `remaining_lease` ↔ `lease_commence_date`: r=0.999 → kept both for tree models (invariant to collinearity), drop one for linear models
- `floor_area_sqm` ↔ `flat_type_enc`: r=0.95 → both kept
- `hdb_age` ↔ `remaining_lease`: r≈−1.0 → both kept (different computation paths)

### Step 3 — Untapped Signal Discovery

Late-stage discovery of high-value unused features:
```
bus_stop_name mean encoding:   r=+0.703  (1,657 unique stops)
full_postal mean encoding:     r=+0.913  (near address-level)
postal_sector mean encoding:   r=+0.538  (245 sectors)
pri_sch_name mean encoding:    r=+0.490  (177 schools)
sec_sch_name mean encoding:    r=+0.486  (134 schools)
```

**However**: Adding these as global mean encodings made LB scores *worse* despite better OOF scores — due to compounding leakage. CatBoost handles all of these natively without any manual encoding.

### Step 4 — Features DROPPED and Why

| Feature | Reason Dropped |
|---|---|
| `block` (raw) | High cardinality string; captured via `address` |
| `street_name` (raw) | Same as block; merged into `address` |
| `storey_range` (raw string) | Parsed into numeric `mid_storey` |
| `floor_area_sqft` | Duplicate of `floor_area_sqm` (× 10.764) |
| `addr_mean_price` | Global encoding — causes leakage in LGB/XGB |
| `town_mean_price` | Same leakage issue |
| `Tranc_YearMonth` | Decomposed into `Tranc_Year` + `Tranc_Month` |
| `1room_rental`–`other_room_rental` | Corr < 0.08, adds noise |
| `vacancy` | Corr = −0.016, no predictive value |
| `hawker_market_stalls` | Corr = −0.009, noise |
| `precinct_pavilion` | Near-zero importance across all models |

---

## 🤖 Model Evolution

### V1 — LightGBM + XGBoost Baseline
```python
# Global mean encoding
addr_mean_price = train.groupby('address')['resale_price'].mean()

# Fixed blend
final = 0.50 * lgb_preds + 0.50 * xgb_preds
```
**LB: 22,309** — set the benchmark

### V2 — OOF Mean Encoding
```python
# Per-fold encoding (honest, no leakage)
fold_addr_mean = tr_df.groupby('address')['resale_price'].mean()
val_addr = val_df['address'].map(fold_addr_mean).fillna(fold_global)
```
**LB: 22,382** — worse than global. OOF encoding was too conservative.

### V3–V4 — More Features + More Trees
- Added room composition ratios (`pct_large`, `pct_3room`)  
- Added `full_flat_type` mean encoding  
- Increased to 20,000 trees, early stopping 500  
**LB: 22,378–22,394** — marginal differences, plateau reached

### V5–V6 — Returning to Global Encoding + More Encodings
- V5: Back to global encoding, 70 features → **LB: 22,379**  
- V6: Added 6 more mean encodings (bus stop, postal, schools) → **LB: 22,459** (worse!)

**Lesson**: More mean encodings compound leakage on the test set. Each additional global encoding adds training-time signal that evaporates on truly unseen data.

### V7 — CatBoost (WINNER 🏆)
```python
# No manual encoding at all
CAT_COLS = ['town','flat_type','flat_model','planning_area',
            'mrt_name','full_flat_type','address']

# CatBoost handles encoding internally via ordered boosting
train_pool = Pool(X_train, y_train, cat_features=cat_indices)
cb = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=8)
cb.fit(train_pool, eval_set=val_pool)
```
**LB: 21,615** — beat Team 1 by 62 points 🏆

---

## 🐱 Why CatBoost Won

### 1. Ordered Target Encoding (Zero Leakage)

CatBoost computes target statistics for each training example using **only the examples that appeared before it** in a random permutation — similar in spirit to OOF encoding but applied at the individual row level:

```
For row i with address "173, Yishun Ave 7":
  CatBoost encodes = avg(resale_price for all j < i with same address)
  
  → Each row's encoding is computed from different data
  → Zero information leakage from validation rows
  → OOF RMSE ≈ true generalisation performance
```

Compare this to global mean encoding where every row (including validation) has seen all prices for its address.

### 2. Symmetric Trees (Oblivious Decision Trees)

LightGBM and XGBoost build asymmetric trees that can overfit to local patterns. CatBoost builds **symmetric (oblivious) trees** where every node at the same depth uses the same split condition:

```
LGB/XGB asymmetric tree:          CatBoost symmetric tree:
        [mrt < 500]                    [mrt < 500]
       /           \                  /           \
  [area > 90]   [floor > 8]      [area > 90]   [area > 90]
  /    \         /    \           /    \         /    \
 ...   ...     ...   ...        ...   ...     ...   ...
```

This makes CatBoost:
- **More regularised** by default (harder to memorise training data)
- **Faster at inference** (lookup table evaluation)
- **More robust** to the high-cardinality `address` feature (9,157 values)

### 3. Native Handling of High-Cardinality Categoricals

For a feature like `address` with 9,157 unique values:

| Method | LGB/XGB approach | CatBoost approach |
|---|---|---|
| Label encoding | 1–9,157 ordinal | N/A |
| Mean encoding (global) | Avg price per address | N/A |
| Mean encoding (OOF) | Avg from other folds | N/A |
| **Native** | ❌ not available | ✅ ordered target stat |

CatBoost extracts **far more information** from `address`, `mrt_name`, and `full_flat_type` than any manual encoding — as shown by `full_flat_type` becoming the **#1 feature at 14.32% importance**, something it never achieved with label encoding in LGB/XGB.

### 4. The OOF–Leaderboard Gap Comparison

| Model | OOF RMSE | LB RMSE | Gap | Verdict |
|---|---|---|---|---|
| LGB+XGB global | 21,610 | 22,379 | +769 | Severe leakage |
| LGB+XGB OOF | 22,461 | 22,378 | −83 | Honest but weak |
| **CatBoost** | **21,769** | **21,615** | **−154** | **Honest + strong** ✅ |

CatBoost's leaderboard score was **154 points better** than OOF — meaning it generalises *beyond* what training validation suggests. This is the hallmark of a well-regularised model that hasn't overfit.

---

## 🎛️ Hyperparameters

### Final CatBoost Configuration

```python
CB_PARAMS = dict(
    iterations          = 10000,   # Max boosting rounds
    learning_rate       = 0.03,    # Step size (CB converges faster than LGB at same LR)
    depth               = 8,       # Symmetric tree depth
    l2_leaf_reg         = 3.0,     # L2 regularisation (penalises large leaf weights)
    random_strength     = 1.0,     # Randomisation for split scoring (prevents overfit)
    bagging_temperature = 1.0,     # Bayesian bootstrap row sampling intensity
    border_count        = 128,     # Number of candidate splits per numeric feature
    od_type             = 'Iter',  # Overfitting detector: stop after od_wait iterations
    od_wait             = 200,     # Early stopping patience
    random_seed         = 42,
    task_type           = 'GPU',   # GPU acceleration
    eval_metric         = 'RMSE',
    loss_function       = 'RMSE',
)
```

### Tuning Guide

| Parameter | Effect | Range to Try |
|---|---|---|
| `iterations` | More trees = better fit (with early stopping) | 10,000–20,000 |
| `learning_rate` | Lower = better generalisation, slower | 0.01–0.05 |
| `depth` | Deeper = more complex patterns | 6–10 |
| `l2_leaf_reg` | Higher = more regularisation | 1–10 |
| `random_strength` | Higher = more randomisation | 0.5–2.0 |
| `bagging_temperature` | Higher = more diverse trees | 0.5–2.0 |
| `od_wait` | Higher = more patience before stopping | 100–500 |

### Why LR=0.03 vs LR=0.01 (used for LGB/XGB)

CatBoost's ordered boosting introduces inherent variance reduction per tree — it effectively gets "more" out of each iteration than LGB/XGB. LR=0.03 with 10,000 trees for CatBoost is approximately equivalent to LR=0.01 with 15,000 trees for LightGBM in terms of convergence quality.

---

## 📚 Key Lessons Learned

### ✅ What Worked

1. **CatBoost native encoding** — eliminated leakage entirely, improved LB by 694 points over our best LGB/XGB score
2. **Log-transforming the target** — `log1p(resale_price)` stabilised training and reduced sensitivity to outliers
3. **Interaction features** — `lease_x_area` became the #2 feature in CatBoost at 7.24%
4. **Distance log-transforms** — `log_mrt_dist` better captures the non-linear proximity premium
5. **5-fold CV** — essential for stable OOF estimates and averaging test predictions
6. **Early stopping with patience** — prevented overfitting, identified true optimal iterations

### ❌ What Didn't Work

1. **Global mean encoding for LGB/XGB** — inflated OOF RMSE by ~750 points vs leaderboard. The model learned to trust `addr_mean_price` but that signal weakened on truly unseen addresses
2. **Adding more mean encodings** — each additional encoding (bus stop, postal, school name) compounded leakage. V6 with 11 encodings scored *worse* (22,459) than V5 with 5 (22,379)
3. **More features ≠ better results** — going from 43 → 70 → 77 features with LGB/XGB consistently hurt leaderboard scores despite better OOF metrics
4. **Addr price slope feature** — year-on-year appreciation per address was noisy for addresses with <3 transactions, adding noise to OOF
5. **Increasing trees beyond convergence** — LGB/XGB had genuinely converged at ~700–1,200 iterations; pushing to 20,000 yielded <2 RMSE improvement
6. **Random Forest** — not tested due to time, but likely would have helped as a third ensemble model

### 🔍 The Core Insight

> **The gap between OOF RMSE and leaderboard RMSE is your leakage detector.**
> 
> LGB/XGB versions: OOF always ~750 points better than LB → severe leakage  
> CatBoost V7: LB **154 points better** than OOF → genuine generalisation  
>
> When your model scores better on the leaderboard than your validation, you've built something that truly generalises.

---

## 🔄 How to Reproduce

### Requirements
```bash
pip install catboost lightgbm xgboost scikit-learn pandas numpy
```

### Run the Winning Model (Kaggle)

1. Upload `train.csv`, `test.csv`, `sample_sub_reg.csv` to a Kaggle dataset named `hdb-data`
2. Create a new Kaggle notebook and paste `kaggle_catboost_v7.py`
3. Enable GPU T4 x2 accelerator
4. Run All → download `submission_kaggle_v7.csv`
5. Submit to competition

### Run Locally

```python
# Set paths in kaggle_catboost_v7.py
BASE_PATH = './data'           # folder containing train.csv, test.csv
USE_GPU   = False              # set True if CUDA GPU available
OUTPUT    = './submissions/submission_v7.csv'
```

### Expected Runtime

| Environment | Time |
|---|---|
| Kaggle GPU T4 x2 | ~25 minutes |
| Local GPU (RTX 3080+) | ~20 minutes |
| Local CPU (8 cores) | ~3–4 hours |

---

## 📈 Leaderboard Progression

```
22,309  ████████████████████████████████████████ submission_official (LGB+XGB)
22,394  █████████████████████████████████████████ submission_v3
22,382  ████████████████████████████████████████ submission_oof
22,378  ████████████████████████████████████████ submission_v4
22,379  ████████████████████████████████████████ submission_v5
22,459  █████████████████████████████████████████ submission_v6 (peaked wrong direction)
21,615  ████████████████████████████████████ submission_v7 CatBoost 🏆
```

---

## 📖 References

1. Wang, L., Chan, F.F., Wang, Y., & Chang, Q. (2016). *Predicting Public Housing Prices Using Delayed Neural Networks*. IEEE TENCON 2016.

2. Liu, Z., & Pan, Y. (2024). *Research of the Influence Factors of Housing Price — Take Singapore as an Example*. Proceedings of ICFTBA 2024.

3. CatBoost documentation — Ordered Target Encoding: https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic

4. Singapore Housing Development Board — Resale Flat Prices: https://data.gov.sg/collections/189/view

---

## 📝 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with ❤️ for the Kaggle Regression Challenge (HDB Price)*  
*Final Score: **21,615.51 RMSE — 1st Place** 🥇*
