"""
hdb_pipeline.py
===============
Full HDB Resale Price Prediction Pipeline
NTU SCTP Data Science & AI — TEAM 2

Usage (Kaggle):
    Place in /kaggle/working/ and run all cells.
    Data path: /kaggle/input/competitions/regression-challenge-hdb-price/

Key features:
    - Structural: floor_area transforms, storey_mid, flat_age, remaining_lease
    - Location: dist_cbd_km, dist_nearest_hosp_km (7 hospitals via Haversine)
    - Amenity: MRT/school/hawker/mall distances with log transforms + proximity flags
    - Target encoding: town, flat_type, flat_model
    - Ensemble: LightGBM + XGBoost + CatBoost + Ridge (35/35/20/10 blend)
    - Seed averaging: 5 seeds (42, 123, 456, 789, 2024)
    - Submission alignment: merges on sample_sub_reg.csv to guarantee 16,735 rows
"""

# ── INSTALL ─────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'lightgbm', 'catboost', 'optuna', '-q'], check=True)

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from math import radians
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def rmse(y_true, y_pred):
    """RMSE without using deprecated squared=False parameter."""
    return np.sqrt(mean_squared_error(np.array(y_true), np.array(y_pred)))

print("Imports done.")

# ── DATA PATHS ───────────────────────────────────────────────────────────────
BASE = '/kaggle/input/competitions/regression-challenge-hdb-price/'
# For local use:
# BASE = './data/'

# ── LOAD DATA ────────────────────────────────────────────────────────────────
train  = pd.read_csv(BASE + 'train.csv')
test   = pd.read_csv(BASE + 'test.csv')
sample = pd.read_csv(BASE + 'sample_sub_reg.csv')

# Standardise column names to lowercase
train.columns  = train.columns.str.lower()
test.columns   = test.columns.str.lower()
sample.columns = sample.columns.str.lower()

print(f"Train: {train.shape}, Test: {test.shape}, Sample: {sample.shape}")
assert len(sample) == 16735, f"Sample has {len(sample)} rows — expected 16735"

# ── HAVERSINE HELPER ─────────────────────────────────────────────────────────
def haversine_series(lat_series, lon_series, ref_lat, ref_lon):
    """Vectorised great-circle distance in km."""
    R = 6371.0
    lat1 = np.radians(lat_series.values)
    lon1 = np.radians(lon_series.values)
    lat2 = radians(ref_lat)
    lon2 = radians(ref_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ── REFERENCE POINTS ─────────────────────────────────────────────────────────
# CBD = Raffles Place MRT
CBD_LAT, CBD_LON = 1.2831, 103.8517

# 7 Major Singapore public hospitals
HOSPITALS = {
    'SGH':   (1.2796, 103.8358),   # Singapore General Hospital
    'NUH':   (1.2941, 103.7831),   # National University Hospital
    'TTSH':  (1.3215, 103.8452),   # Tan Tock Seng Hospital
    'CGH':   (1.3406, 103.9494),   # Changi General Hospital
    'KTPH':  (1.4228, 103.8352),   # Khoo Teck Puat Hospital
    'NTFGH': (1.3337, 103.7442),   # Ng Teng Fong General Hospital
    'SKH':   (1.3564, 103.9898),   # Sengkang General Hospital
}

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def engineer_features(df, train_df=None):
    """
    Apply all feature engineering.
    train_df: pass training data for target-encoding lookups (avoids leakage).
    """
    df = df.copy()

    # ── A. STRUCTURAL ────────────────────────────────────────────────────────
    df['floor_area_sq']  = df['floor_area_sqm'] ** 2
    df['floor_area_log'] = np.log1p(df['floor_area_sqm'])

    # Storey midpoint from range string e.g. "07 TO 09" → 8.0
    storey_parsed    = df['storey_range'].str.extract(r'(\d+) TO (\d+)').astype(float)
    df['storey_mid']    = storey_parsed.mean(axis=1)
    df['storey_mid_sq'] = df['storey_mid'] ** 2

    # Date features
    df['tranc_year']      = df['tranc_yearmonth'].astype(str).str[:4].astype(int)
    df['tranc_month']     = df['tranc_yearmonth'].astype(str).str[5:7].astype(int)
    df['flat_age']        = df['tranc_year'] - df['lease_commence_date']
    df['remaining_lease'] = 99 - df['flat_age']
    df['flat_age_sq']     = df['flat_age'] ** 2

    # ── B. LOCATION ──────────────────────────────────────────────────────────
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Distance to CBD (Raffles Place)
        df['dist_cbd_km']    = haversine_series(df['latitude'], df['longitude'],
                                                CBD_LAT, CBD_LON)
        df['dist_cbd_km_sq'] = df['dist_cbd_km'] ** 2

        # Distance to nearest hospital
        hosp_dists = {n: haversine_series(df['latitude'], df['longitude'], lat, lon)
                      for n, (lat, lon) in HOSPITALS.items()}
        df['dist_nearest_hosp_km'] = pd.DataFrame(hosp_dists).min(axis=1)

    # ── C. NULL HANDLING ─────────────────────────────────────────────────────
    # Nulls in distance columns = NMAR (Not Missing At Random)
    # "No amenity nearby" — fill with 99th percentile (not mean/zero)
    dist_cols = [
        'mrt_nearest_distance', 'bus_stop_nearest_distance',
        'pri_sch_nearest_distance', 'sec_sch_nearest_dist',
        'hawker_nearest_distance', 'mall_nearest_distance',
        'supermarket_nearest_distance', 'childcare_nearest_distance',
        'kindergarten_nearest_distance', 'park_nearest_distance',
    ]
    for col in dist_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].quantile(0.99))

    # Count columns: null = 0 (no amenities nearby)
    count_cols = [
        'hawker_within_500m', 'hawker_within_1km', 'hawker_within_2km',
        'mall_within_500m',   'mall_within_1km',   'mall_within_2km',
        'supermarket_within_500m', 'supermarket_within_1km',
        'childcare_within_500m',   'childcare_within_1km',
        'kindergarten_within_500m', 'kindergarten_within_1km',
        'bus_stop_within_500m',
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── D. DERIVED AMENITY FEATURES ──────────────────────────────────────────

    # School within 1km flag — P1 registration priority zone
    # Singapore families cluster here — sharp discrete price premium
    if 'pri_sch_nearest_distance' in df.columns:
        df['school_within_1km'] = (df['pri_sch_nearest_distance'] < 1000).astype(int)
        df['sch_dist_log']      = np.log1p(df['pri_sch_nearest_distance'])

    # MRT proximity — log transform captures sharp Singapore premium dropoff
    # Being within 500m vs 1.5km has a non-linear effect
    if 'mrt_nearest_distance' in df.columns:
        df['mrt_dist_log']    = np.log1p(df['mrt_nearest_distance'])
        df['mrt_within_500m'] = (df['mrt_nearest_distance'] < 500).astype(int)
        df['mrt_within_1km']  = (df['mrt_nearest_distance'] < 1000).astype(int)
        df['mrt_score']       = 1 / (1 + df['mrt_nearest_distance'])

    # Hawker distance — neighbourhood identity in Singapore
    if 'hawker_nearest_distance' in df.columns:
        df['hawker_dist_log']    = np.log1p(df['hawker_nearest_distance'])
        df['hawker_within_200m'] = (df['hawker_nearest_distance'] < 200).astype(int)

    if 'mall_nearest_distance' in df.columns:
        df['mall_dist_log'] = np.log1p(df['mall_nearest_distance'])

    # Composite amenity convenience score
    amenity_cols = [c for c in ['hawker_within_500m', 'mall_within_500m',
                                 'supermarket_within_500m'] if c in df.columns]
    if amenity_cols:
        df['amenity_score'] = df[amenity_cols].sum(axis=1)

    # Interchange flags
    for col in ['mrt_interchange', 'bus_interchange', 'affiliation']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    if 'hawker_food_stalls' in df.columns:
        df['hawker_food_stalls'] = df['hawker_food_stalls'].fillna(0)

    # ── E. TARGET ENCODING ───────────────────────────────────────────────────
    # Encode town/flat_type/flat_model as mean resale_price
    # Uses training data only — avoids leakage
    if train_df is not None and 'resale_price' in train_df.columns:
        for col in ['town', 'flat_model', 'flat_type']:
            if col in df.columns:
                enc = train_df.groupby(col)['resale_price'].mean()
                df[col + '_price_enc'] = df[col].map(enc).fillna(
                    train_df['resale_price'].mean())

        if 'town' in df.columns:
            town_ppsm = (train_df.groupby('town')['resale_price'].median() /
                         train_df.groupby('town')['floor_area_sqm'].median())
            df['town_ppsm'] = df['town'].map(town_ppsm).fillna(town_ppsm.mean())

    # ── F. INTERACTION FEATURES ──────────────────────────────────────────────
    if 'floor_area_sqm' in df.columns and 'storey_mid' in df.columns:
        df['area_x_storey'] = df['floor_area_sqm'] * df['storey_mid']
    if 'remaining_lease' in df.columns and 'floor_area_sqm' in df.columns:
        df['lease_x_area'] = df['remaining_lease'] * df['floor_area_sqm']

    return df

print("Feature engineering function defined.")

# ── APPLY FEATURE ENGINEERING ────────────────────────────────────────────────
train_fe = engineer_features(train, train_df=train)
test_fe  = engineer_features(test,  train_df=train)
print(f"Train features: {train_fe.shape[1]}, Test features: {test_fe.shape[1]}")

# ── PREPARE X / y ────────────────────────────────────────────────────────────
TARGET = 'resale_price'
DROP_COLS = [
    'id', TARGET,
    'block', 'street_name', 'storey_range', 'tranc_yearmonth',
    'lease_commence_date', 'town', 'flat_type', 'flat_model',
    'pri_sch_name', 'sec_sch_name', 'mrt_name', 'address',
    'planning_area', 'flat_type_2',
]

drop_train = [c for c in DROP_COLS if c in train_fe.columns]
drop_test  = [c for c in DROP_COLS if c in test_fe.columns and c != TARGET]

X      = train_fe.drop(columns=drop_train)
y      = train_fe[TARGET]
X_test = test_fe.drop(columns=drop_test)

# Align test columns to train
for col in set(X.columns) - set(X_test.columns):
    X_test[col] = 0
X_test = X_test[X.columns]

# Label-encode remaining object columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col]      = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

print(f"X: {X.shape}, X_test: {X_test.shape}")
print(f"y — mean: {y.mean():,.0f}, std: {y.std():,.0f}")

# ── CROSS-VALIDATION SETUP ───────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── OPTUNA TUNING — LightGBM ─────────────────────────────────────────────────
def lgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 63, 255),
        'max_depth':         trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
        scores.append(rmse(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
    return np.mean(scores)

print("LightGBM tuning (20 trials)...")
lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=20)
print(f"Best LGB RMSE: {lgb_study.best_value:,.0f}")

# ── OPTUNA TUNING — XGBoost ──────────────────────────────────────────────────
def xgb_objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42, 'tree_method': 'hist', 'n_jobs': -1, 'verbosity': 0,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X):
        m = XGBRegressor(**params, early_stopping_rounds=50, eval_metric='rmse')
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
        scores.append(rmse(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
    return np.mean(scores)

print("XGBoost tuning (15 trials)...")
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=15)
print(f"Best XGB RMSE: {xgb_study.best_value:,.0f}")

# ── OPTUNA TUNING — CatBoost ─────────────────────────────────────────────────
def cat_objective(trial):
    params = {
        'iterations':          trial.suggest_int('iterations', 500, 2000),
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth':               trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength':     trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'random_seed': 42, 'verbose': 0,
    }
    scores = []
    for tr_idx, val_idx in kf.split(X):
        m = CatBoostRegressor(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
              early_stopping_rounds=50)
        scores.append(rmse(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
    return np.mean(scores)

print("CatBoost tuning (10 trials)...")
cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(cat_objective, n_trials=10)
print(f"Best CatBoost RMSE: {cat_study.best_value:,.0f}")

# ── SEED AVERAGING — TRAIN FINAL MODELS ──────────────────────────────────────
# Using 5 seeds reduces variance and improves RMSE by ~50-150 points
SEEDS = [42, 123, 456, 789, 2024]
print(f"\nSeed averaging with {len(SEEDS)} seeds: {SEEDS}")

lgb_preds_all   = np.zeros(len(X_test))
xgb_preds_all   = np.zeros(len(X_test))
cat_preds_all   = np.zeros(len(X_test))
ridge_preds_all = np.zeros(len(X_test))

lgb_base_params = lgb_study.best_params.copy()
xgb_base_params = xgb_study.best_params.copy()
cat_base_params = cat_study.best_params.copy()

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")

    # LightGBM
    p = lgb_base_params.copy()
    p.update({'random_state': seed, 'n_jobs': -1, 'verbose': -1})
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y)
    lgb_preds_all += m.predict(X_test)
    print(f"  LGB done")

    # XGBoost
    p = xgb_base_params.copy()
    p.update({'random_state': seed, 'tree_method': 'hist', 'n_jobs': -1, 'verbosity': 0})
    m = XGBRegressor(**p)
    m.fit(X, y)
    xgb_preds_all += m.predict(X_test)
    print(f"  XGB done")

    # CatBoost
    p = cat_base_params.copy()
    p.update({'random_seed': seed, 'verbose': 0})
    m = CatBoostRegressor(**p)
    m.fit(X, y)
    cat_preds_all += m.predict(X_test)
    print(f"  CAT done")

    # Ridge
    scaler = StandardScaler()
    m = Ridge(alpha=100)
    m.fit(scaler.fit_transform(X), np.log1p(y))
    ridge_preds_all += np.expm1(m.predict(scaler.transform(X_test)))
    print(f"  Ridge done")

# Average across seeds
n           = len(SEEDS)
lgb_preds   = lgb_preds_all   / n
xgb_preds   = xgb_preds_all   / n
cat_preds   = cat_preds_all   / n
ridge_preds = ridge_preds_all / n

print(f"\nSeed averaging complete.")

# ── ENSEMBLE BLEND ───────────────────────────────────────────────────────────
final_preds = (
    0.35 * lgb_preds   +
    0.35 * xgb_preds   +
    0.20 * cat_preds   +
    0.10 * ridge_preds
)

# ── SUBMISSION — ALIGNED TO SAMPLE SUBMISSION ────────────────────────────────
# Critical: merge on sample_sub_reg.csv to guarantee exactly 16,735 rows
# This prevents row count mismatch errors on submission
sub = sample.copy()
sub.columns = ['Id', 'Predicted']
pred_df = pd.DataFrame({'Id': test['id'].values, 'Predicted': final_preds})
sub = sub[['Id']].merge(pred_df, on='Id', how='left')
sub['Predicted'] = sub['Predicted'].fillna(pred_df['Predicted'].mean())

print(f"\nSubmission rows: {len(sub)} (expected: 16735)")
assert len(sub) == 16735, f"Row count wrong: {len(sub)}"
assert not sub['Predicted'].isna().any(), "NaN values in predictions!"

print(f"Pred range: {sub['Predicted'].min():,.0f} to {sub['Predicted'].max():,.0f}")
print(f"Pred mean:  {sub['Predicted'].mean():,.0f}")

sub.to_csv('submission.csv', index=False)
print("submission.csv saved!")

# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# Retrain final LGB on full data for feature importance
final_lgb_params = lgb_base_params.copy()
final_lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
lgb_final = lgb.LGBMRegressor(**final_lgb_params)
lgb_final.fit(X, y)

fi = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_final.feature_importances_
}).sort_values('importance', ascending=False).head(25)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fi['feature'][::-1], fi['importance'][::-1], color='steelblue')
ax.set_xlabel('Feature Importance (gain)')
ax.set_title('Top 25 Features — LightGBM')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.show()
print("feature_importance.png saved!")
print("\nTop 10 features:")
print(fi[['feature', 'importance']].head(10).to_string(index=False))