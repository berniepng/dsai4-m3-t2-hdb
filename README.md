# 🏠 HDB Resale Price Prediction — Full Competition Journey

> **Competition**: Regression Challenge (HDB Resale Price Prediction)  
> **Metric**: RMSE (Root Mean Squared Error) — lower is better  
> **Team**: TEAM 2  
> **Best Score**: 21,471.62 (2nd place on public leaderboard)  
> **Model**: CatBoost ensemble with native categorical encoding

---

## 📋 Table of Contents

1. [What We Were Trying to Do](#what-we-were-trying-to-do)
2. [The Dataset](#the-dataset)
3. [Our Journey — Version by Version](#our-journey)
4. [Key Turning Points](#key-turning-points)
5. [Feature Engineering Deep Dive](#feature-engineering-deep-dive)
6. [Why CatBoost Beat Everything Else](#why-catboost-beat-everything-else)
7. [Challenges and How We Managed Them](#challenges-and-how-we-managed-them)
8. [Model Limitations](#model-limitations)
9. [Key Learnings](#key-learnings)
10. [What We Would Do With No Constraints](#what-we-would-do-with-no-constraints)
11. [How to Reproduce](#how-to-reproduce)

---

## 🎯 What We Were Trying to Do

Predict the **resale price of HDB flats in Singapore** as accurately as possible, given a transaction's physical attributes, location, and timing.

Think of it like this: a buyer looks at a flat and asks *"what is this worth?"*. Our model answers that question using 63 measurable signals — from floor area and storey level to which primary school is nearby and how far the nearest MRT station is.

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

| Property | Detail |
|---|---|
| Source | Singapore HDB via Kaggle competition |
| Training rows | 150,634 transactions (2012–2021) |
| Test rows | 16,735 transactions |
| Features | 77 columns per transaction |
| Target | `resale_price` (SGD) |
| Price range | $150,000 – $1,258,000 |
| Median price | $420,000 |

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
└─────────────────────────────────────────────────────────────┘
```

**Important note**: The competition prohibited external datasets — every feature had to come from the provided data files. This constraint shaped many of our decisions.

---

## 🗺️ Our Journey — Version by Version

### The Score Progression

```
22,500 ┤
       │  V1    V2    V3    V4    V5    V6
22,400 ┤  ●─────●─────●─────●─────●─────●
       │                                   \
22,300 ┤                                    ●  (V1 best: 22,309)
       │
22,200 ┤
       │
22,100 ┤
       │
22,000 ┤
       │
21,900 ┤
       │                                         V7 CatBoost
21,800 ┤                                         ●
       │                                          \
21,700 ┤                                           \  V8
       │                                            ●
21,600 ┤                                             \  V11
       │                                              ●
21,500 ┤                                               \  V12  Blend
       │                                                ●──────●
21,400 ┤                                                    Group 9 ●
       │
```

---

### Phase 1 — Wrong Data Source (Before V1)

**What happened**: We initially downloaded raw CSV files from data.gov.sg, the Singapore government's open data portal. These files contained basic transaction data but were missing the rich location features (MRT distances, school data, mall proximity) that the competition's official training set had.

**Why it mattered**: We were trying to predict prices without the most important signals. Like trying to value a flat without knowing its neighbourhood.

**How we fixed it**: Discovered the official `train.csv` from Kaggle had 77 enriched features vs the raw gov.sg files' ~15. Switched entirely to the official dataset.

**Lesson**: Always verify your data source matches the competition's test set structure.

---

### Phase 2 — LightGBM + XGBoost Baseline (V1–V6)

**What we built**: A standard machine learning ensemble using two popular gradient boosting models — LightGBM (LGB) and XGBoost (XGB) — blended together.

**Gradient boosting explained simply**: Imagine building a team of 500 specialists. Each specialist focuses on correcting the mistakes of the previous one. The final prediction is everyone's combined opinion. LGB and XGB are different implementations of this idea.

```
V1 approach:
  addr_mean_price = average(resale_price) per address  ← computed from ALL data
  Features: 43 total (physical + location + distances)
  Models: LGB + XGB blended 50/50
  LB Score: 22,309
```

**The leakage problem discovered**: Our `addr_mean_price` feature was computed using the entire training set — including the validation rows we were trying to predict. This created "data leakage" — the model was peeking at the answer.

```
❌ WRONG (global encoding):
   addr_mean_price for "173 Yishun Ave 7" 
   = average of ALL transactions at that address
   = validation rows' prices are included!
   = artificially inflated training score

✅ RIGHT (OOF encoding):
   addr_mean_price for fold 3 validation
   = average of folds 1,2,4,5 transactions only
   = validation rows never seen during encoding
```

**What we tried**: Switching to OOF (Out-of-Fold) encoding, adding more features (up to 77), adding more mean encodings (bus stops, schools, postal codes). None of it helped — the leaderboard score stubbornly stayed around 22,300–22,459.

**The uncomfortable truth**: More features with LGB/XGB made things *worse*. V6 with 11 encodings scored 22,459 — the worst result of the LGB/XGB era.

| Version | Features | Encoding | LB Score |
|---|---|---|---|
| V1 | 43 | Global (leaked) | 22,309 ✅ best LGB |
| V2 | 43 | OOF (honest) | 22,382 |
| V3 | 60 | OOF | 22,394 |
| V4 | 70 | OOF, 20K trees | 22,378 |
| V5 | 70 | Global | 22,379 |
| V6 | 77 | Global + 11 enc | 22,459 ❌ worst |

---

### Phase 3 — The CatBoost Breakthrough (V7)

**The insight**: LightGBM and XGBoost cannot handle categorical text features (like town names, flat types, MRT station names) natively. They require manual encoding — which either leaks data (global) or loses signal (OOF). CatBoost was designed to handle this problem differently.

**What CatBoost does differently — Ordered Target Encoding**:

```
Standard mean encoding (LGB/XGB):
  "WOODLANDS" → $380,000  ← average of ALL Woodlands transactions
                            validation rows included = LEAKAGE

CatBoost ordered encoding:
  For row 5,000 in "WOODLANDS":
  → Uses only rows 1–4,999 that are also "WOODLANDS"
  → Each row sees only transactions that came BEFORE it
  → Zero leakage by construction
```

**Result**:

```
V7: CatBoost, 45 features, 7 raw categoricals
    OOF RMSE:  21,769  (honest)
    LB Score:  21,615  ← LB BETTER than OOF!

    Compare to V5: OOF 21,610 (leaked) → LB 22,379 (800 worse)
    CatBoost gap: −154 points (model generalises beyond training)
    LGB/XGB gap: +769 points (leakage inflates training score)
```

**This was the single biggest turning point.** Switching from LGB/XGB to CatBoost improved the leaderboard score by **694 points** (22,309 → 21,615) — equivalent to predicting each flat $694 more accurately on average.

**Feature importance from CatBoost V7**:

```
full_flat_type    ████████████████  14.3%  ← #1 (was never #1 in LGB/XGB)
floor_area_sqm    █████████          8.6%
lease_x_area      ███████            7.2%
dist_cbd          ██████             6.9%
time_index        ██████             6.6%
mrt_name          █████              5.1%
town              █████              5.0%
```

---

### Phase 4 — Seed Ensembling (V8)

**The concept**: Training the same model 3 times with different random seeds produces slightly different predictions. Averaging them reduces random noise.

```
Seed 42   predicts flat X: $452,000
Seed 2024 predicts flat X: $448,000
Seed 888  predicts flat X: $449,000
─────────────────────────────────────
Ensemble average:          $449,667  ← smoother, more accurate
```

**V8**: 3 seeds × 10 folds = 30 models  
**OOF gain from ensembling**: +159 RMSE points  
**LB Score**: 21,615 (V7) → not directly comparable (V8 used different hyperparams)

---

### Phase 5 — Region Features + Learning Rate Fix (V11)

**New features**: Singapore's property market is officially segmented into three regions:
- **CCR** (Core Central Region): Districts 9, 10, 11 — prime, highest prices
- **RCR** (Rest of Central Region): Districts 1–8, 12–15, 20 — mid-tier
- **OCR** (Outside Central Region): Districts 16–28 — heartland HDB majority

We added `region`, `region × flat_type`, and `region × town` as raw categoricals.

**Critical bug fixed**: V10 (not shown) used `iterations=25,000` with `learning_rate=0.03`. Every single fold hit the iteration cap without ever triggering early stopping — the model kept memorising training data past its optimal point. V11 fixed this with `learning_rate=0.02`.

```
V10 (broken):   best_iter = 24,997  ← hit cap, never converged
V11 (fixed):    best_iter = 11,303  ← stopped at right time ✅
```

**LB Score**: 21,499 — improvement of 116 points over V8 equivalent.

---

### Phase 6 — School Prestige Features (V12)

**The real-world insight**: Singapore has a strict 1km priority rule for primary school registration. Families pay a premium to live within 1km of prestigious schools (GEP schools, SAP schools). This is a well-known property market phenomenon locally.

**What we added**:
- `pri_sch_name` and `sec_sch_name` as raw CatBoost categoricals (177 + 134 schools)
- `school_prestige` tier (1–5) based on MOE GEP/SAP rankings from Skoolopedia
- `prestige_in_1km`: school tier × within-1km flag
- Non-linear transforms: `floor_area²`, `storey²`, `remaining_lease²`
- 5 seeds instead of 3 (50 total models)

**Price premium discovered**:
```
Near Pei Hwa Presbyterian (Tier 5 GEP school): avg $790,503
Near non-prestigious school:                    avg $446,000
Premium:                                        +$344,503 (+77%)
```

**Result**: LB 21,494 — only 5 points better than V11. The school signals were already partially captured by `address`, `town`, and `planning_area`.

---

### Phase 7 — Blending (Final)

**The simple idea**: Average the predictions from V11 and V12. Where the two models disagree, the truth often lies between them.

```
522 rows where V11 and V12 disagreed by >$5,000:
  Example: Flat ID 58743
  V11 predicted: $879,168
  V12 predicted: $913,714
  Blend (50/50): $896,441  ← splits the difference
```

**Result**: blend_50_50 scored **21,471** — the biggest single jump since V7:
- From V12 (21,494) to blend (21,471): **−23 points**
- Achieved **2nd place** on the public leaderboard

---

## 🔑 Key Turning Points

### Turning Point 1: Discovering the Data Leakage (V1→V7)
**Impact: +694 RMSE points**

The moment we realised that global mean encoding was inflating our training score by ~750 points — and that CatBoost eliminates this problem natively — was the most important insight of the competition.

```
LGB/XGB global encoding:  OOF 21,610 → LB 22,379  (gap: +769)
CatBoost ordered:         OOF 21,769 → LB 21,615  (gap: -154) ✅
```

### Turning Point 2: Switching to CatBoost (V6→V7)
**Impact: +694 RMSE on leaderboard**

Not just a model swap — a fundamentally different approach to encoding categorical data.

### Turning Point 3: Seed Ensembling (V7→V8)
**Impact: +159 OOF RMSE from averaging**

Averaging 30 models reduced prediction variance significantly.

### Turning Point 4: Blending V11 + V12 (V12→Blend)
**Impact: +23 LB RMSE, jumped from #3 to #2**

Two independently trained model families, averaged together, reduced noise further.

---

## ⚙️ Feature Engineering Deep Dive

### Features We Kept and Why

**Time features** — property prices follow market cycles:
```python
time_index = (Tranc_Year - 2012) × 12 + Tranc_Month
# Continuous time signal capturing market cycles
# Most important feature across all versions
```

**Physical interactions** — size × storey is non-linear:
```python
area_x_storey    = floor_area_sqm × mid_storey
lease_x_area     = remaining_lease × floor_area_sqm
floor_area_sq    = floor_area_sqm²   # non-linear size premium
storey_sq        = mid_storey²       # high floors exponentially valuable
```

**Location at 6 granularities** (from finest to coarsest):
```
address (9,157 unique)    → block-level
full_postal (9,124 unique) → postal code
bus_stop_name (1,657)     → hyper-local
postal_sector (245)       → 3-digit postal
planning_area (32)        → sub-region
town (26)                 → HDB town
```

**School prestige** — exploiting Singapore's 1km rule:
```python
school_prestige = {
    'Nanyang Primary School': 5,    # GEP school
    'Henry Park Primary School': 5, # GEP, affiliated Hwa Chong
    'Nan Hua Primary School': 4,    # SAP school
    ...
}
prestige_in_1km = school_prestige × (distance ≤ 1000m)
```

### Features We Dropped and Why

| Feature | Reason Dropped |
|---|---|
| `floor_area_sqft` | Exact duplicate of `floor_area_sqm` × 10.764 |
| `lease_commence_date` | Near-perfect correlation with `remaining_lease` (r=0.999) |
| `flat_model` | VIF > 5 — collinear with other features (Liu & Pan, 2024) |
| `1room_rental` to `other_room_rental` | Correlation < 0.08 — noise |
| `addr_price_slope` | Noisy for addresses with < 3 transactions |
| `storey_range` (raw string) | Parsed into numeric `mid_storey` |

---

## 🐱 Why CatBoost Beat Everything Else

### The Leakage Gap — Visualised

```
What LGB/XGB sees:                What CatBoost sees:
─────────────────                 ──────────────────
addr_mean_price                   No manual encoding needed
= avg of ALL rows                 → learns internally using
  including validation            → ordered statistics
  ← CHEATING!                    → zero leakage ✅

OOF score:  21,610 (optimistic)   OOF score:  21,769 (honest)
LB score:   22,379 (reality)      LB score:   21,615 (reality)
Gap:        +769 points           Gap:        -154 points
```

### Symmetric Trees

LightGBM and XGBoost grow asymmetric trees that can overfit local patterns. CatBoost grows **symmetric (oblivious) trees** — every node at the same depth uses the same split condition. This makes CatBoost:
- More regularised by default
- Better at handling high-cardinality features like `address` (9,157 unique values)
- More stable across different random seeds

### The Numbers Don't Lie

| Model | LB Score | vs CatBoost |
|---|---|---|
| LGB + XGB (best) | 22,309 | +694 worse |
| **CatBoost V7** | **21,615** | **baseline** |

---

## 🚧 Challenges and How We Managed Them

### Challenge 1: Session Timeouts

**Problem**: Training 50 CatBoost models takes 7+ hours. Our development environment (Claude.ai sandbox) resets between sessions, losing all trained models.

**Solution**: Migrated entirely to Kaggle notebooks. Kaggle provides free GPU (T4 × 2) with 12-hour session limits and persistent output storage. Used "Save & Run All (Commit)" so training runs server-side even with the laptop closed.

### Challenge 2: Wrong Dataset

**Problem**: Initial data sourced from data.gov.sg had only ~15 features vs the competition's 77-feature enriched dataset. We were building models on incomplete data.

**Solution**: Discovered the official `train.csv` from the Kaggle competition page had all enriched features pre-computed (MRT distances, school data, etc.). Rebuilt everything from scratch.

### Challenge 3: The OOF vs Leaderboard Gap

**Problem**: Our LGB/XGB models showed great OOF scores (~21,500) but leaderboard scores were ~22,300 — a 800-point gap that didn't make sense initially.

**Solution**: Identified global mean encoding as the cause. When `addr_mean_price` is computed from all training data including validation rows, the model "cheats" during validation. On the test set (truly unseen), this leakage provides no benefit, so performance collapses. CatBoost eliminated this entirely.

```
The telltale sign of leakage:
OOF score << LB score  →  leakage present
OOF score ≈ LB score   →  honest model
OOF score >> LB score  →  model generalises well (ideal)
```

### Challenge 4: The Problem Fold (Seed 42, Fold 4)

**Problem**: Across every version, one specific fold consistently scored ~22,399 — over 1,000 points worse than other folds. This dragged the seed OOF higher.

**Root cause**: Standard KFold's random split happened to cluster rare, expensive flats (CCR executive units, multi-generation) into the validation set of Fold 4. The model hadn't seen enough of these during training to predict them accurately.

**Solution implemented**: V13 switches to Stratified KFold, splitting the data into 10 price quantile bands and ensuring each fold has the same proportion of cheap, mid-range, and expensive flats.

```
Standard KFold (bad):        Stratified KFold (V13):
Fold 4 validation:           Fold 4 validation:
  45% budget flats    ❌       10% budget flats     ✅
  35% mid-range       ❌       10% mid-range        ✅
  20% expensive       ❌       10% expensive        ✅
  → model unprepared          → model well prepared
```

### Challenge 5: The Data Leakage Patch Backfired

**Problem**: We discovered 523 test rows were exact duplicates of training rows — meaning we knew their prices. A patch script was written to override model predictions with known prices.

**Result**: Leaderboard score got *worse* (21,853 vs 21,494).

**Why it backfired**: CatBoost's ordered encoding had already seen these rows in training folds 80% of the time — it was already predicting them accurately. The patch introduced noise because some duplicates had multiple conflicting prices in training ($303,000 vs $318,000 for the same flat in the same month).

**Lesson**: Don't patch what isn't broken. Verify your assumptions before overriding a well-trained model.

---

## ⚠️ Model Limitations

### 1. Temporal Generalisation
The model was trained on 2012–2021 data. Post-2022 market conditions (interest rate rises, cooling measures, COVID aftermath) are not represented. Predictions for 2024–2025 transactions would likely be less accurate.

### 2. No Renovation or Condition Data
HDB resale prices are heavily influenced by renovation quality, interior condition, and furniture. A flat in original 1990s condition vs one with a full $80,000 renovation can differ by $100,000+ at the same address. The dataset has no proxy for this.

### 3. The 30/70 Public/Private Split
The current leaderboard uses only 30% of test data. Final rankings shift when the remaining 70% is scored. Our 2nd-place position may change — positively or negatively.

### 4. View and Facing Direction
A flat facing a reservoir vs facing another HDB block can command a 10–15% premium. `Latitude` and `Longitude` partially proxy this but not precisely.

### 5. The Problem Fold
Seed 42 Fold 4 consistently underperforms. This suggests a systematic data distribution issue — likely rare flat combinations clustering in one validation split — that standard KFold cannot handle.

### 6. Competition Data Constraint
No external data was allowed. In a real-world setting, adding PropertyGuru listing data, URA private transaction comparables, or interest rate history would materially improve predictions.

---

## 📚 Key Learnings

### 1. Data Leakage Is the Biggest Risk in ML Competitions
**The OOF vs leaderboard gap is your leakage detector.**
- OOF much better than LB → leakage in training
- OOF ≈ LB → honest model
- LB better than OOF → genuinely generalising (ideal, rare)

### 2. Model Choice Matters More Than Feature Engineering
Switching from LGB/XGB to CatBoost gained **694 points**. Adding 34 more features to LGB/XGB gained **0 points**. Sometimes the right tool matters more than more features.

### 3. Categorical Features Need Native Handling
Manual encoding (label encoding, mean encoding) always involves a tradeoff between leakage and information loss. CatBoost's ordered encoding eliminates both issues simultaneously.

### 4. Ensembling is Free Money
- Multiple seeds → +176 OOF RMSE points (V12 ensemble gain)
- Blending V11 + V12 → +23 LB RMSE points
- These gains require zero new features or model complexity

### 5. The OOF–LB Gap Reveals Your Model's Integrity

| Model | OOF | LB | Gap | Interpretation |
|---|---|---|---|---|
| V5 global LGB | 21,610 | 22,379 | +769 | Severe leakage |
| V7 CatBoost | 21,769 | 21,615 | −154 | Genuine generalisation |
| V12 CatBoost | 21,324 | 21,494 | +170 | Slight leakage (school enc) |

### 6. Simple Blends Often Beat Complex Features
Adding school prestige features (hours of work) gained 5 LB points. Blending two existing submissions (minutes of work) gained 23 LB points. Always try the simple thing first.

### 7. Convergence Must Be Verified
V10's catastrophic failure (OOF 21,554 vs V8's 21,416) happened because every fold hit the iteration cap — early stopping never triggered. Always verify `best_iter` is well below `max_iterations`.

```
Good convergence:    best_iter = 11,303  (max: 20,000) ✅
Bad convergence:     best_iter = 24,997  (max: 25,000) ❌
```

---

## 🚀 What We Would Do With No Constraints

Given unlimited data, compute, and time — here's how to build a materially better model:

### 1. External Data Sources

| Data | Signal | Expected Gain |
|---|---|---|
| PropertyGuru listing prices | Days on market, asking vs selling price | High |
| URA private transaction data | Private condo comparables nearby | High |
| HDB resale price index (quarterly) | Market cycle timing | Medium |
| Interest rate history (MAS) | Mortgage affordability shifts | Medium |
| Google Street View images | Block condition, neighbourhood quality | High |
| School ballot results (annual) | Actual oversubscription rate | Medium |

### 2. Renovation and Condition Proxy
Scrape PropertyGuru listings for HDB flats, extract renovation keywords (e.g. "renovated 2022", "original condition", "full reno") and build a renovation quality score. This alone could explain $50,000–$100,000 in price variance.

### 3. Spatial Features
Compute distances to:
- Hawker centres with specific cuisine type ratings
- Parks and green corridors (within 500m)
- Community centres, polyclinics, wet markets
- Specific MRT lines (Circle Line commands different premium than North-South)

### 4. View and Orientation
Using satellite imagery or building footprint data, determine:
- Whether a flat faces a reservoir, sea, or park
- Whether it faces another block (corridor/privacy issue)
- Floor-to-ceiling height estimation from building plans

### 5. Time-Aware Models
Use a time-series aware cross-validation — train on 2012–2018, validate on 2019, then train 2012–2019, validate on 2020, etc. This would:
- Prevent future data leaking into past predictions
- Better estimate how the model performs on future transactions
- Allow the model to learn from market cycle patterns explicitly

### 6. Neural Architecture
For a dataset this size (150K rows), a **Gradient Boosted Tree + Neural Network blend** would capture both:
- Tabular structure (trees handle this well)
- Complex non-linear interactions (neural networks handle this better)

Specifically, **FT-Transformer** (Feature Tokenizer + Transformer) has shown strong results on tabular regression tasks of this scale.

### 7. Stacking with a Meta-Learner
Instead of a simple average blend:
```
Level 1: CatBoost OOF predictions
Level 1: LightGBM OOF predictions  → Ridge regression → Final prediction
Level 1: XGBoost OOF predictions
```
This learns the optimal blend weights from data rather than fixing them manually.

---

## 📈 Full Competition Leaderboard History

| Version | Model | Key Change | LB Score | Position |
|---|---|---|---|---|
| V1 (submission_official) | LGB+XGB | Baseline, global encoding | 22,309 | #2 |
| V2 (submission_oof) | LGB+XGB | OOF encoding | 22,382 | — |
| V3 | LGB+XGB | 60 features | 22,394 | — |
| V4 | LGB+XGB | 70 feat, 20K trees | 22,378 | — |
| V5 | LGB+XGB | Global enc, 70 feat | 22,379 | — |
| V6 | LGB+XGB | 77 feat, 11 encodings | 22,459 | — |
| **V7** | **CatBoost** | **Native encoding, 45 feat** | **21,615** | **🥇 #1** |
| V8 | CatBoost | 3-seed×10-fold, tuned | 21,538 | — |
| V11 | CatBoost | Region features, LR fix | 21,499 | #3 |
| V12 | CatBoost | School names, 5 seeds | 21,494 | #3 |
| **Blend 50/50** | **V11+V12** | **Equal blend** | **21,471** | **🥈 #2** |
| **Blend 45/55** | **V11+V12** | **V12-favoured** | **21,471** | **🥈 #2** |
| V13 (pending) | CatBoost | Stratified KFold | TBD | TBD |

---

## 🔄 How to Reproduce

### Requirements
```bash
pip install catboost lightgbm xgboost scikit-learn pandas numpy
```

### Run the Winning Model (Kaggle)

1. Upload `train.csv`, `test.csv`, `sample_sub_reg.csv` to a Kaggle dataset
2. Create a new Kaggle notebook
3. Set accelerator: **GPU T4 × 2**
4. Paste `kaggle_catboost_v12.py` and update `BASE_PATH` to your dataset path
5. **Save Version → Save & Run All (Commit)**
6. Download `submission_kaggle_v12.csv` from Output tab
7. Submit to competition

### Expected Runtime

| Environment | Time |
|---|---|
| Kaggle GPU T4 × 2 | ~7 hours (50 models) |
| Local GPU (RTX 3080+) | ~5 hours |
| Local CPU (8 cores) | ~30–40 hours |

---

## 📖 Academic References

1. Wang, L., Chan, F.F., Wang, Y., & Chang, Q. (2016). *Predicting Public Housing Prices Using Delayed Neural Networks*. IEEE TENCON 2016.
   - Key finding: Floor area, floor level, MRT distance, lease duration are top HDB price signals

2. Liu, Z., & Pan, Y. (2024). *Research of the Influence Factors of Housing Price — Take Singapore as an Example*. ICFTBA 2024.
   - Key finding: Flat model and lease commence date are collinear (r=0.999) — drop one
   - MLR R² = 0.674 with 4 clean features (floor area, flat type, storey, remaining lease)

3. Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS 2018.
   - The paper describing CatBoost's ordered encoding — the core of our winning approach

---

*Built for the Kaggle Regression Challenge (HDB Resale Price Prediction)*  
*Public Leaderboard: 2nd Place — 21,471.62 RMSE*
