# ============================================================
# HDB RESALE PRICE PREDICTION — KAGGLE NOTEBOOK V23
#
# Base: V22 (80 features — V16 + block composition)
# New:  +2 room prestige features, cleaner flat_type signal = 82 total
#
# ============================================================
# PROBLEM WITH V22 FLAT_TYPE ENCODING
# ============================================================
# V22 (and all prior versions) encodes flat_type as a linear
# ordinal 1→7 via FLAT_MAP:
#   1 ROOM=1, 2 ROOM=2, 3 ROOM=3, 4 ROOM=4,
#   5 ROOM=5, EXECUTIVE=6, MULTI-GENERATION=7
#
# This is NOISY for two reasons:
#
#   1. MULTI-GENERATION (n≈77) is ultra-rare and sits at 7 —
#      dragging the linear relationship and injecting noise
#      at the extreme. Price is NOT monotonically higher than
#      EXECUTIVE for this flat type.
#
#   2. The ordinal spacing (1,2,3,4,5,6,7) implies equal price
#      increments between room types, but the real market has
#      a non-linear jump between 4-ROOM and 5-ROOM/EXECUTIVE.
#      The top tier commands a structural premium beyond what
#      linear spacing captures.
#
# ============================================================
# V23 FIX: ROOM PRESTIGE TIER (2 NEW FEATURES)
# ============================================================
#
#   room_prestige — Cleaner signal than flat_type_enc:
#       5 ROOM        → 3  (premium tier)
#       EXECUTIVE     → 3  (premium tier — same as 5 ROOM)
#       4 ROOM        → 2  (standard tier — middle market)
#       1/2/3 ROOM    → 1  (economy tier)
#       MULTI-GEN     → 1  (lumped with economy — avoids noise)
#       Rationale: 5 ROOM and EXECUTIVE command similar price
#       premiums and are both "large flat" types. MULTI-GEN
#       is too rare (n=77) to deserve its own bin — grouping
#       it with economy removes the noisy 7-point outlier.
#
#   is_premium_flat — Binary flag (5 ROOM or EXECUTIVE = 1)
#       Interaction with block_quality and location.
#       A 5-ROOM in a premium block ≠ a 5-ROOM in a rental
#       block. This binary captures the "large flat" premium
#       cleanly without the MULTI-GEN outlier problem.
#       Expected correlation: r≈+0.55 (similar to flat_type)
#       but cleaner because it collapses the noisy top end.
#
# NOTE: flat_type_enc and n_rooms are KEPT — the original
#   ordinal signal is still useful for CatBoost's ordered
#   encoding. The new features ADD information; they don't
#   replace the originals. Total: +2 features.
#
# ============================================================
# WHAT STAYS (V22 features — all proven)
# ============================================================
#   Block composition (V22 new, all timeless):
#     pct_premium, pct_economy, pct_rental, has_rental,
#     has_commercial, block_quality
#   Planning maturity (V16):
#     planning_tier, is_mature_estate, pt_x_flattype
#   School demand (V16):
#     oversubscription, oversub_in_1km, is_top_school_1km
#   MOP dynamics (V15):
#     mop_just_passed, is_mop_window, years_past_mop
#   Era-relative size (V15):
#     area_vs_era, area_vs_era_pct, storey_vs_era
#   Cooling measures (V15):
#     5 × mths_{year}_{month} + months_since_last_cooling
#
# ============================================================
# ARCHITECTURE: identical to V22/V16 (proven best)
#   CatBoost: 5 seeds × 10 stratified folds = 50 models
#   LightGBM: 3 seeds × 10 KFold = 30 models
#   Total: 80 models, auto-optimised blend weight from OOF
#
# KILL/GO THRESHOLDS (CB Seed 1 = seed 42):
#   Fold 2 < 20,950  → very strong, definitely GO
#   Fold 2 < 21,050  → solid improvement, likely GO
#   Fold 2 > 21,200  → concerning, watch Fold 3-4
#   Seed 1 OOF < 21,200  → GO — genuine improvement on V22
#   Seed 1 OOF < 21,300  → marginal — similar to V22, run all seeds
#   Seed 1 OOF > 21,350  → KILL — worse than V16 baseline
#
#   TARGET: Seed 1 OOF < 21,300 to beat V22/V16 OOF of ~21,300
#   STRETCH: Blend V23+V11 to beat our best LB of 21,394
#
# OOF ARRAYS: all 5 saved for ridge stacking
# Features: 82 total (70 numeric + 12 categorical)
#   V22: 80 | +room prestige: +2 = 82
# ============================================================

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

# ============================================================
# 1. SETTINGS
# ============================================================
DATASET   = 'hdb-data'
BASE_PATH = f'/kaggle/input/datasets/posclaude666/{DATASET}'
OUTPUT    = '/kaggle/working/submission_kaggle_v23.csv'
USE_GPU   = True
N_FOLDS   = 10
CB_SEEDS  = [42, 2024, 888, 123, 9999]
LGB_SEEDS = [42, 2024, 888]
W_CB_DEFAULT = 0.75

DISTRICT_REGION = {
    1:'RCR',2:'RCR',3:'RCR',4:'RCR',5:'OCR',6:'RCR',7:'RCR',8:'RCR',
    9:'CCR',10:'CCR',11:'CCR',12:'RCR',13:'RCR',14:'RCR',15:'RCR',
    16:'OCR',17:'OCR',18:'OCR',19:'OCR',20:'RCR',21:'OCR',22:'OCR',
    23:'OCR',24:'OCR',25:'OCR',26:'OCR',27:'OCR',28:'OCR'
}
MATURE_ESTATES = {
    'ANG MO KIO','BEDOK','BISHAN','BUKIT MERAH','BUKIT TIMAH','CENTRAL AREA',
    'CLEMENTI','GEYLANG','KALLANG/WHAMPOA','MARINE PARADE','PASIR RIS',
    'QUEENSTOWN','SERANGOON','TAMPINES','TOA PAYOH'
}
PLANNING_TIER = {
    'CENTRAL AREA':5,'QUEENSTOWN':5,'BISHAN':5,'BUKIT TIMAH':5,'TOA PAYOH':5,'MARINE PARADE':5,
    'BUKIT MERAH':4,'KALLANG/WHAMPOA':4,'CLEMENTI':4,'SERANGOON':4,'ANG MO KIO':4,'GEYLANG':4,
    'BEDOK':3,'TAMPINES':3,'PASIR RIS':3,'HOUGANG':3,'SENGKANG':3,'PUNGGOL':3,
    'WOODLANDS':2,'YISHUN':2,'JURONG EAST':2,'JURONG WEST':2,'CHOA CHU KANG':2,
    'BUKIT BATOK':2,'BUKIT PANJANG':2,'SEMBAWANG':2
}
OVERSUBSCRIPTION = {
    'Nanyang Primary School':5,'Henry Park Primary School':5,'Ai Tong School':5,
    'Catholic High School':5,'Pei Hwa Presbyterian Primary School':5,'Tao Nan School':5,
    'Rosyth School':5,'Raffles Girls Primary School':5,'Anglo-Chinese School (Primary)':5,
    'Methodist Girls School':5,'Nan Hua Primary School':4,'Rulang Primary School':4,
    'Kong Hwa School':4,'Fairfield Methodist School':4,'Pei Chun Public School':4,
    'Maris Stella High School':4,'Queenstown Primary School':4,'Nan Chiau Primary School':4,
    'Red Swastika School':4,'Poi Ching School':4,'River Valley Primary School':4,
    'Chongfu School':4,'CHIJ Our Lady of Good Counsel':4,'De La Salle School':4,
    'Cantonment Primary School':3,'Saint Hilda\'s Primary School':3,
    'Bedok Green Primary School':3,'Kuo Chuan Presbyterian Primary School':3,
    'Geylang Methodist School':3,'Holy Innocents\' Primary School':3,
}
COOLING_DATES = [(2011,12),(2013,1),(2013,6),(2018,7),(2021,12)]

# V22 original ordinal map — kept for continuity
FLAT_MAP = {'1 ROOM':1,'2 ROOM':2,'3 ROOM':3,'4 ROOM':4,
            '5 ROOM':5,'EXECUTIVE':6,'MULTI-GENERATION':7}

# V23 NEW: Clean room prestige tier
# Collapses MULTI-GENERATION (n=77, noisy outlier) into economy tier
# Groups 5 ROOM and EXECUTIVE as same "large flat" premium tier
# This removes the false linearity of 1,2,3,4,5,6,7 spacing
ROOM_PRESTIGE = {
    '5 ROOM':3,            # Premium tier
    'EXECUTIVE':3,         # Premium tier (same tier as 5 ROOM)
    '4 ROOM':2,            # Standard tier
    '3 ROOM':1,            # Economy tier
    '2 ROOM':1,            # Economy tier
    '1 ROOM':1,            # Economy tier
    'MULTI-GENERATION':1,  # Lumped with economy — avoids n=77 noise
}
PREMIUM_FLAT_TYPES = {'5 ROOM', 'EXECUTIVE'}

CAT_COLS  = ['town','flat_type','flat_model','planning_area','mrt_name',
             'full_flat_type','address','region','region_x_flattype',
             'region_x_town','pri_sch_name','sec_sch_name']

# ============================================================
# 2. DATA LOADING
# ============================================================
print("Loading data...", flush=True)
try:
    train = pd.read_csv(f'{BASE_PATH}/train.csv', low_memory=False)
    print(f"  Loaded: train.csv — {len(train):,} rows")
except FileNotFoundError:
    parts = [pd.read_csv(f'{BASE_PATH}/train_part{i}.csv',low_memory=False) for i in [1,2,3]]
    train = pd.concat(parts, ignore_index=True)
    print(f"  Loaded: train_part1/2/3.csv — {len(train):,} rows")

test   = pd.read_csv(f'{BASE_PATH}/test.csv',           low_memory=False)
sample = pd.read_csv(f'{BASE_PATH}/sample_sub_reg.csv', low_memory=False)

train['resale_price'] = pd.to_numeric(train['resale_price'])
train['address'] = train['block'].astype(str).str.strip()+', '+train['street_name'].str.strip()
test['address']  = test['block'].astype(str).str.strip()+', '+test['street_name'].str.strip()
train['Tranc_Year']  = pd.to_numeric(train['Tranc_Year'],  errors='coerce').astype(int)
train['Tranc_Month'] = pd.to_numeric(train['Tranc_Month'], errors='coerce').astype(int)
test['Tranc_Year']   = pd.to_numeric(test['Tranc_Year'],   errors='coerce').astype(int)
test['Tranc_Month']  = pd.to_numeric(test['Tranc_Month'],  errors='coerce').astype(int)
print(f"  Train: {len(train):,} | Test: {len(test):,}")

# Print V23 room prestige distribution for verification
print("\n  V23 Room Prestige Distribution:")
rp = train['flat_type'].map(ROOM_PRESTIGE).fillna(1)
for tier, name in [(3,'Premium (5R+Exec)'),(2,'Standard (4R)'),(1,'Economy (1-3R+MG)')]:
    n = (rp==tier).sum()
    pct = n/len(train)*100
    print(f"    Tier {tier} {name}: {n:,} ({pct:.1f}%)")

# ============================================================
# 3. PRE-COMPUTE SHARED STATS (train only — no leakage)
# ============================================================
print("\nPre-computing shared statistics...", flush=True)

for c in ['floor_area_sqm','mid_storey','lease_commence_date','Latitude','Longitude']:
    train[c] = pd.to_numeric(train[c], errors='coerce')
    test[c]  = pd.to_numeric(test[c],  errors='coerce')

train['decade_built'] = (train['lease_commence_date']//10*10)
test['decade_built']  = (test['lease_commence_date']//10*10)
cohort_area   = train.groupby(['flat_type','decade_built'])['floor_area_sqm'].median()
cohort_storey = train.groupby(['flat_type','decade_built'])['mid_storey'].median()

# ── Block composition features (V22 — timeless structural signals) ──
print("  Computing block composition features...")
for df in [train, test]:
    for c in ['1room_sold','2room_sold','3room_sold','4room_sold','5room_sold',
              'exec_sold','multigen_sold','studio_apartment_sold',
              '1room_rental','2room_rental','3room_rental','other_room_rental',
              'total_dwelling_units']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    total = df['total_dwelling_units'].clip(lower=1)

    df['pct_premium']   = (df['5room_sold'] + df['exec_sold']) / total
    df['pct_economy']   = (df['1room_sold'] + df['2room_sold'] + df['3room_sold']) / total
    df['total_rental']  = (df['1room_rental'] + df['2room_rental'] +
                            df['3room_rental'] + df['other_room_rental'])
    df['pct_rental']    = df['total_rental'] / total
    df['has_rental']    = (df['total_rental'] > 0).astype(float)
    df['has_commercial']= (df['commercial'].astype(str)=='Y').astype(float)
    df['block_quality'] = df['pct_premium'] - df['pct_economy'] - df['pct_rental']

# Report V22 block composition correlations
print("  Block composition feature correlations:")
for feat in ['pct_premium','pct_economy','pct_rental','has_rental',
             'has_commercial','block_quality']:
    r = float(train[feat].corr(train['resale_price']))
    print(f"    {feat:<20}: r={r:+.4f}")

# ── Base enrichment ───────────────────────────────────────────
def enrich_base(df):
    df = df.copy()
    df['postal_num']       = pd.to_numeric(df['postal'], errors='coerce')
    df['postal_district']  = df['postal_num'].apply(
        lambda x: int(str(int(x))[:2]) if pd.notna(x) and x>0 else 0)
    df['postal_sector']    = df['postal_num'].apply(
        lambda x: int(str(int(x))[:3]) if pd.notna(x) and x>0 else 0)
    df['region']           = df['postal_district'].map(DISTRICT_REGION).fillna('OCR')
    df['region_x_flattype']= df['region']+'_'+df['flat_type'].astype(str)
    df['region_x_town']    = df['region']+'_'+df['town'].astype(str)
    df['planning_tier']    = df['town'].map(PLANNING_TIER).fillna(2.0)
    df['is_mature_estate'] = df['town'].isin(MATURE_ESTATES).astype(float)
    df['oversubscription'] = df['pri_sch_name'].map(OVERSUBSCRIPTION).fillna(0.0)
    return df

train = enrich_base(train)
test  = enrich_base(test)

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\nBuilding feature matrix...", flush=True)

def build_features(df, cohort_area, cohort_storey):
    f  = pd.DataFrame(index=df.index)
    df = df.copy()
    for c in ['floor_area_sqm','mid_storey','max_floor_lvl','hdb_age',
              'lease_commence_date','total_dwelling_units','Latitude','Longitude',
              'mrt_nearest_distance','Mall_Nearest_Distance','Hawker_Nearest_Distance',
              'bus_stop_nearest_distance','pri_sch_nearest_distance']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['decade_built'] = (df['lease_commence_date']//10*10)

    # ── Time ─────────────────────────────────────────────────
    f['Tranc_Year']    = df['Tranc_Year'].astype(int)
    f['Tranc_Month']   = df['Tranc_Month'].astype(int)
    f['time_index']    = (df['Tranc_Year']-2012)*12 + df['Tranc_Month']

    # ── Physical ─────────────────────────────────────────────
    f['floor_area_sqm']      = df['floor_area_sqm']
    f['floor_area_sq']       = df['floor_area_sqm']**2
    f['mid_storey']          = df['mid_storey']
    f['storey_sq']           = df['mid_storey']**2
    f['max_floor_lvl']       = df['max_floor_lvl']
    f['storey_ratio']        = df['mid_storey']/(df['max_floor_lvl']+1)
    f['hdb_age']             = df['hdb_age']
    f['lease_commence_date'] = df['lease_commence_date']
    f['remaining_lease']     = (99-(df['Tranc_Year']-df['lease_commence_date'])).clip(0,99)
    f['remaining_lease_sq']  = f['remaining_lease']**2
    f['total_dwelling_units']= df['total_dwelling_units']

    # ── V22 original flat type ordinal (kept for CatBoost ordered encoding) ──
    f['flat_type_enc']       = df['flat_type'].map(FLAT_MAP).fillna(4)
    f['n_rooms']             = df['flat_type'].map(FLAT_MAP).fillna(4)
    f['sqm_per_room']        = df['floor_area_sqm']/f['n_rooms']
    f['multistorey_carpark'] = (df['multistorey_carpark']=='Y').astype(int)

    # ── V23 NEW: Room prestige tier ───────────────────────────
    # Replaces noisy 1-7 ordinal with cleaner 3-tier signal:
    # Economy(1-3R+MG)=1 | Standard(4R)=2 | Premium(5R+Exec)=3
    # Eliminates MULTI-GENERATION outlier (n=77) noise
    # Collapses 5-ROOM and EXECUTIVE into same premium tier
    f['room_prestige']       = df['flat_type'].map(ROOM_PRESTIGE).fillna(1).astype(float)
    f['is_premium_flat']     = df['flat_type'].isin(PREMIUM_FLAT_TYPES).astype(float)

    # ── Interactions ─────────────────────────────────────────
    f['area_x_storey']       = df['floor_area_sqm']*df['mid_storey']
    f['area_per_storey']     = df['floor_area_sqm']/(df['mid_storey']+1)
    f['lease_x_area']        = f['remaining_lease']*df['floor_area_sqm']
    f['age_x_lease']         = df['hdb_age']*f['remaining_lease']

    # V23 NEW: Premium flat × block quality interaction
    # A 5-ROOM in a premium block commands a different premium
    # than a 5-ROOM in a rental-heavy block
    f['premium_x_blockquality'] = f['is_premium_flat'] * df['block_quality']

    # ── MOP (V15) ────────────────────────────────────────────
    f['mop_just_passed'] = ((df['hdb_age']>=5)&(df['hdb_age']<=8)).astype(float)
    f['is_mop_window']   = ((df['hdb_age']>=5)&(df['hdb_age']<=10)).astype(float)
    f['years_past_mop']  = (df['hdb_age']-5).clip(lower=0)

    # ── Era-relative size (V15) ───────────────────────────────
    key = list(zip(df['flat_type'].astype(str), df['decade_built']))
    era_a = pd.Series([cohort_area.get(k,np.nan) for k in key],
                      index=df.index).fillna(df['floor_area_sqm'].median())
    era_s = pd.Series([cohort_storey.get(k,np.nan) for k in key],
                      index=df.index).fillna(df['mid_storey'].median())
    f['area_vs_era']     = df['floor_area_sqm'] - era_a
    f['area_vs_era_pct'] = f['area_vs_era'] / era_a
    f['storey_vs_era']   = df['mid_storey'] - era_s

    # ── Cooling measures (V15) ────────────────────────────────
    t = df['Tranc_Year'] + (df['Tranc_Month']-1)/12
    since_cols = []
    for yr,mo in COOLING_DATES:
        mt  = yr + (mo-1)/12
        col = f'mths_{yr}_{mo:02d}'
        f[col] = ((t-mt)*12).clip(lower=0).where(t>=mt, 0)
        since_cols.append(col)
    f['months_since_last_cooling'] = f[since_cols].apply(
        lambda row: min([v for v in row if v>0], default=0), axis=1)

    # ── Planning area maturity (V16) ─────────────────────────
    f['planning_tier']    = df['planning_tier'].astype(float)
    f['is_mature_estate'] = df['is_mature_estate'].astype(float)
    f['pt_x_flattype']    = df['planning_tier']*df['flat_type'].map(FLAT_MAP).fillna(4)

    # ── School oversubscription (V16) ─────────────────────────
    pri_dist = df['pri_sch_nearest_distance']
    f['oversubscription']  = df['oversubscription'].astype(float)
    f['oversub_in_1km']    = df['oversubscription']*(pri_dist<=1000).astype(float)
    f['is_top_school_1km'] = ((df['oversubscription']>=4)&(pri_dist<=1000)).astype(float)

    # ── Block composition (V22 — timeless) ───────────────────
    f['pct_premium']    = df['pct_premium'].fillna(0)
    f['pct_economy']    = df['pct_economy'].fillna(0)
    f['pct_rental']     = df['pct_rental'].fillna(0)
    f['has_rental']     = df['has_rental'].fillna(0)
    f['has_commercial'] = df['has_commercial'].fillna(0)
    f['block_quality']  = df['block_quality'].fillna(0)

    # ── Location ─────────────────────────────────────────────
    f['postal_district']  = df['postal_district'].astype(float)
    f['postal_sector']    = df['postal_sector'].astype(float)
    f['Latitude']         = df['Latitude']
    f['Longitude']        = df['Longitude']
    f['dist_cbd']         = np.sqrt((df['Latitude']-1.2837)**2+(df['Longitude']-103.8517)**2)*111

    # ── MRT ──────────────────────────────────────────────────
    f['mrt_nearest_distance'] = df['mrt_nearest_distance']
    f['log_mrt_dist']         = np.log1p(df['mrt_nearest_distance'])
    f['mrt_interchange']      = pd.to_numeric(df['mrt_interchange'],errors='coerce').fillna(0)
    f['bus_interchange']      = pd.to_numeric(df['bus_interchange'],errors='coerce').fillna(0)
    f['bus_stop_nearest']     = df['bus_stop_nearest_distance']

    # ── Mall ─────────────────────────────────────────────────
    f['Mall_Nearest_Distance'] = df['Mall_Nearest_Distance']
    f['log_mall_dist']         = np.log1p(df['Mall_Nearest_Distance'].fillna(664))
    f['Mall_Within_1km']       = pd.to_numeric(df['Mall_Within_1km'],errors='coerce').fillna(0)
    f['Mall_Within_2km']       = pd.to_numeric(df['Mall_Within_2km'],errors='coerce').fillna(0)

    # ── Hawker ───────────────────────────────────────────────
    f['Hawker_Nearest_Distance'] = df['Hawker_Nearest_Distance']
    f['log_hawker_dist']         = np.log1p(df['Hawker_Nearest_Distance'])
    f['Hawker_Within_1km']       = pd.to_numeric(df['Hawker_Within_1km'],errors='coerce').fillna(0)
    f['Hawker_Within_2km']       = pd.to_numeric(df['Hawker_Within_2km'],errors='coerce').fillna(0)
    f['hawker_food_stalls']      = pd.to_numeric(df['hawker_food_stalls'],errors='coerce')

    # ── Schools ──────────────────────────────────────────────
    f['pri_sch_nearest_distance'] = pri_dist
    f['pri_sch_affiliation']      = pd.to_numeric(df['pri_sch_affiliation'],errors='coerce').fillna(0)
    f['sec_sch_nearest_dist']     = pd.to_numeric(df['sec_sch_nearest_dist'],errors='coerce')
    f['cutoff_point']             = pd.to_numeric(df['cutoff_point'],errors='coerce').fillna(0)
    f['affiliation']              = pd.to_numeric(df['affiliation'],errors='coerce').fillna(0)
    f['amenity_score']            = (
        1/(df['mrt_nearest_distance']/1000+0.1)+
        1/(df['Mall_Nearest_Distance'].fillna(664)/1000+0.1)+
        1/(df['pri_sch_nearest_distance']/1000+0.1))

    # ── Categoricals ─────────────────────────────────────────
    for col in CAT_COLS:
        f[col] = df[col].astype(str)

    # Fill numeric NaN
    num_cols = [c for c in f.columns if c not in CAT_COLS]
    for col in num_cols:
        f[col] = pd.to_numeric(f[col],errors='coerce').fillna(
            pd.to_numeric(f[col],errors='coerce').median())
    return f

Xtr_df = build_features(train, cohort_area, cohort_storey)
Xte_df = build_features(test,  cohort_area, cohort_storey)

all_cols    = Xtr_df.columns.tolist()
cat_indices = [all_cols.index(c) for c in CAT_COLS]
Xcb_all = Xtr_df[all_cols].values
Xcb_t   = Xte_df[all_cols].values
y       = np.log1p(train['resale_price'].values)
num_cols = [c for c in all_cols if c not in CAT_COLS]
Xnum_tr  = Xtr_df[num_cols].values.astype(np.float32)
Xnum_te  = Xte_df[num_cols].values.astype(np.float32)

n_num = len(num_cols); n_cat = len(CAT_COLS)
print(f"  Numeric: {n_num} | Categorical: {n_cat} | Total: {n_num+n_cat}")
print(f"  V22:80 + room_prestige:+1 + is_premium_flat:+1 + premium_x_blockquality:+1 = {n_num+n_cat}")
print(f"  New: room_prestige (3-tier clean) + is_premium_flat (binary)")
print(f"       premium_x_blockquality (interaction: large flat × block design)")

# Report V23 new feature correlations
rp_corr = Xtr_df['room_prestige'].corr(pd.Series(np.expm1(y)))
pf_corr = Xtr_df['is_premium_flat'].corr(pd.Series(np.expm1(y)))
pxb_corr = Xtr_df['premium_x_blockquality'].corr(pd.Series(np.expm1(y)))
print(f"\n  V23 new feature correlations:")
print(f"    room_prestige:           r={rp_corr:+.4f}")
print(f"    is_premium_flat:         r={pf_corr:+.4f}")
print(f"    premium_x_blockquality:  r={pxb_corr:+.4f}")

# ============================================================
# 5. LGB ENCODING SETUP
# ============================================================
les = {}
for col in CAT_COLS:
    all_vals = sorted(set(train[col].astype(str))|set(test[col].astype(str)))
    les[col] = LabelEncoder().fit(all_vals)
global_mean = train['resale_price'].mean()
test_enc_cols = []
for col in CAT_COLS:
    g = train.groupby(col)['resale_price'].mean()
    test_enc_cols.append(test[col].astype(str).map(g).fillna(global_mean).values)
Xlgb_te = np.column_stack([Xnum_te]+test_enc_cols).astype(np.float32)

# ============================================================
# 6. CATBOOST TRAINING
# ============================================================
print(f"\n{'='*62}")
print(f"  CATBOOST: {len(CB_SEEDS)} seeds × {N_FOLDS} folds = {len(CB_SEEDS)*N_FOLDS} models")
print(f"  KILL:  Seed 1 OOF > 21,350 (worse than V16 baseline)")
print(f"  WATCH: Seed 1 OOF 21,250-21,350 (marginal)")
print(f"  GO:    Seed 1 OOF < 21,250 (genuine improvement on V22)")
print(f"  TARGET: Beat V22 OOF — must see < 21,300")
print(f"{'='*62}", flush=True)

cb_params = dict(
    iterations=20000, learning_rate=0.02, depth=9,
    l2_leaf_reg=5.0, random_strength=0.5, bagging_temperature=0.8,
    border_count=254, od_type='Iter', od_wait=400,
    task_type='GPU' if USE_GPU else 'CPU',
    eval_metric='RMSE', loss_function='RMSE', verbose=1000,
)

price_bins    = pd.qcut(y, q=10, labels=False)
cb_seed_oofs  = np.zeros((len(CB_SEEDS), len(train)))
cb_test_preds = np.zeros(len(test))
cb_seed_rmses = []

# V22 reference fold scores for Seed 42 (10-fold stratified)
# UPDATE THESE with actual V22 Seed 42 fold results after V22 run
V22_FOLD_REF = {1:21355,2:20997,3:21264,4:21156,5:21298,
                6:21189,7:21412,8:21334,9:21287,10:21418}
# NOTE: V22_FOLD_REF is initialized with V16 values as placeholder.
# Replace with actual V22 Seed 42 fold results once available.
# V16 Seed 42 OOF: ~21,300

for si, seed in enumerate(CB_SEEDS):
    print(f"\n── CB Seed {si+1}/{len(CB_SEEDS)} (seed={seed}) ──", flush=True)
    cb_params['random_seed'] = seed
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_rmses = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xcb_all, price_bins)):
        tp = Pool(Xcb_all[tr_idx],  y[tr_idx],  cat_features=cat_indices)
        vp = Pool(Xcb_all[val_idx], y[val_idx], cat_features=cat_indices)
        xp = Pool(Xcb_t,                         cat_features=cat_indices)
        cb = CatBoostRegressor(**cb_params)
        cb.fit(tp, eval_set=vp, use_best_model=True)
        cb_seed_oofs[si, val_idx] = cb.predict(vp)
        cb_test_preds += cb.predict(xp)/(N_FOLDS*len(CB_SEEDS))
        rmse = np.sqrt(mean_squared_error(
            np.expm1(y[val_idx]), np.expm1(cb_seed_oofs[si,val_idx])))
        fold_rmses.append(rmse)
        if seed == 42:
            ref  = V22_FOLD_REF.get(fold+1, 21300)
            diff = rmse - ref
            flag = '✅' if diff < -50 else '⚠️' if diff > 100 else '→'
            print(f"  Fold {fold+1:>2}: {rmse:,.0f} | V22:{ref:,} | {diff:>+5,.0f} {flag} | iter:{cb.best_iteration_}", flush=True)
        else:
            print(f"  Fold {fold+1:>2}: {rmse:,.0f} | iter:{cb.best_iteration_}", flush=True)
    seed_rmse = np.sqrt(mean_squared_error(
        np.expm1(y), np.expm1(cb_seed_oofs[si])))
    cb_seed_rmses.append(seed_rmse)
    if seed == 42:
        decision = 'GO ✅' if seed_rmse < 21250 else 'WATCH ⚠️' if seed_rmse < 21350 else 'KILL ❌'
        print(f"  Seed 42 OOF: {seed_rmse:,.0f}  V22:~21,300  → {decision}")
    else:
        print(f"  Seed {seed} OOF: {seed_rmse:,.0f}")

cb_oof      = cb_seed_oofs.mean(axis=0)
cb_oof_rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(cb_oof)))
print(f"\nCatBoost ensemble OOF: {cb_oof_rmse:,.0f}  (V22:~21,300  gain:{21300-cb_oof_rmse:+,.0f})")

# ============================================================
# 7. LIGHTGBM TRAINING
# ============================================================
print(f"\n{'='*62}")
print(f"  LIGHTGBM: {len(LGB_SEEDS)} seeds × {N_FOLDS} folds = {len(LGB_SEEDS)*N_FOLDS} models")
print(f"{'='*62}", flush=True)

lgb_params = dict(
    n_estimators=10000, learning_rate=0.02, num_leaves=255,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbose=-1, device='gpu' if USE_GPU else 'cpu',
)

lgb_seed_oofs  = np.zeros((len(LGB_SEEDS), len(train)))
lgb_test_preds = np.zeros(len(test))
lgb_seed_rmses = []

for si, seed in enumerate(LGB_SEEDS):
    print(f"\n── LGB Seed {si+1}/{len(LGB_SEEDS)} (seed={seed}) ──", flush=True)
    lgb_params['random_state'] = seed
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(Xnum_tr)):
        tr_df, val_df = train.iloc[tr_idx], train.iloc[val_idx]
        fg = tr_df['resale_price'].mean()
        def fold_enc(df):
            enc = []
            for col in CAT_COLS:
                g = tr_df.groupby(col)['resale_price'].mean()
                enc.append(df[col].astype(str).map(g).fillna(fg).values)
            return np.column_stack(enc).astype(np.float32)
        Xtr  = np.column_stack([Xnum_tr[tr_idx],  fold_enc(tr_df)]).astype(np.float32)
        Xval = np.column_stack([Xnum_tr[val_idx], fold_enc(val_df)]).astype(np.float32)
        ytr, yval = y[tr_idx], y[val_idx]
        lm = lgb.LGBMRegressor(**lgb_params)
        lm.fit(Xtr, ytr, eval_set=[(Xval,yval)],
               callbacks=[lgb.early_stopping(200,verbose=False),
                          lgb.log_evaluation(2000)])
        lgb_seed_oofs[si, val_idx] = lm.predict(Xval)
        lgb_test_preds += lm.predict(Xlgb_te)/(N_FOLDS*len(LGB_SEEDS))
        rmse = np.sqrt(mean_squared_error(
            np.expm1(yval), np.expm1(lgb_seed_oofs[si,val_idx])))
        print(f"  Fold {fold+1:>2}: {rmse:,.0f} | iter:{lm.best_iteration_}", flush=True)
    seed_rmse = np.sqrt(mean_squared_error(
        np.expm1(y), np.expm1(lgb_seed_oofs[si])))
    lgb_seed_rmses.append(seed_rmse)
    print(f"  Seed {seed} OOF: {seed_rmse:,.0f}  (V22:~22,068)")

lgb_oof      = lgb_seed_oofs.mean(axis=0)
lgb_oof_rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(lgb_oof)))
print(f"\nLightGBM OOF: {lgb_oof_rmse:,.0f}  (V22:~22,068  gain:{22068-lgb_oof_rmse:+,.0f})")

# ============================================================
# 8. SAVE OOF ARRAYS
# ============================================================
np.save('/kaggle/working/oof_preds_cb_v23.npy',  cb_oof)
np.save('/kaggle/working/oof_preds_lgb_v23.npy', lgb_oof)
np.save('/kaggle/working/oof_y_v23.npy',         y)
np.save('/kaggle/working/test_preds_cb_v23.npy', cb_test_preds)
np.save('/kaggle/working/test_preds_lgb_v23.npy',lgb_test_preds)
print("\nOOF arrays saved ✅")

# ============================================================
# 9. BLEND OPTIMISATION + SUBMISSION
# ============================================================
print(f"\n{'='*62}")
print("  BLEND OPTIMISATION")
print(f"{'='*62}")
cb_e  = np.expm1(cb_oof)-np.expm1(y)
lgb_e = np.expm1(lgb_oof)-np.expm1(y)
am,bm = cb_e-cb_e.mean(), lgb_e-lgb_e.mean()
corr  = float((am*bm).sum()/(((am**2).sum()*(bm**2).sum())**0.5+1e-10))
print(f"  CB OOF:  {cb_oof_rmse:>9,.0f}  (V22:~21,300)")
print(f"  LGB OOF: {lgb_oof_rmse:>9,.0f}  (V22:~22,068)")
print(f"  Diversity: {1-corr:.6f}")
print()
best_rmse, best_w = 999999, W_CB_DEFAULT
for w_cb in np.arange(0.65, 1.01, 0.05):
    blend = w_cb*cb_oof + (1-w_cb)*lgb_oof
    brmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(blend)))
    marker = '  ← optimal' if brmse < best_rmse else ''
    print(f"  CB={w_cb:.0%} LGB={1-w_cb:.0%}: {brmse:>10,.0f}{marker}")
    if brmse < best_rmse: best_rmse, best_w = brmse, w_cb

w_lgb_final = round(1-best_w,2)
final_log   = best_w*cb_test_preds + w_lgb_final*lgb_test_preds
final_preds = np.expm1(final_log).clip(100000,1300000)
sub_all     = pd.DataFrame({'Id':test['id'].values,'Predicted':final_preds.astype(int)})
sub         = sample[['Id']].merge(sub_all,on='Id',how='left')
sub.to_csv(OUTPUT,index=False)

print(f"\n{'='*62}")
print(f"  BLEND OOF:  {best_rmse:>9,.0f}  (V22:~21,222  gain:{21222-best_rmse:+,.0f})")
print(f"  Weights:    {best_w:.0%} CB + {w_lgb_final:.0%} LGB")
print(f"{'='*62}")
print(f"\n  Rows:{len(sub):,}  Mean:${final_preds.mean():>10,.0f}")
print(f"\n  Median by flat type:")
m = test.merge(sub,left_on='id',right_on='Id')
for ft,p in m.groupby('flat_type')['Predicted'].median().sort_values().items():
    print(f"    {ft:<22}: ${p:>10,.0f}")
print(f"\n{'='*62}")
print(f"  NEXT STEPS:")
print(f"  1. Download oof_preds_cb_v23.npy + submission_kaggle_v23.csv")
print(f"  2. Blend V23 + V11 (50/50) — proven strategy: ~21,050 LB target")
print(f"  3. Ridge stack V23 OOFs with V22/V16 OOFs for multi-version blend")
print(f"  4. V23+V11 blend desc: V23(82f: V22+room_prestige_tier+is_premium_flat)")
print(f"{'='*62}")
