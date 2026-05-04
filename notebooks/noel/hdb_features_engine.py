# %% [code]
# %% [code]
# %% [code]
import pandas as pd
import numpy as np

def build_features(df):
    logs = []
    
    print ("build_features...", flush=True)
    logs.append("build_features...")
    """
    Consolidated Feature Engineering from Section 4.
    This acts as the 'Contract' between training and inference.
    """
    f = pd.DataFrame(index=df.index)
    
    # ── Time Features ───────────────────────────────────────
    # We assume 'Tranc_Year' and 'Tranc_Month' exist in the raw data
    print ("build_features: Time Features...", flush=True)
    logs.append("build_features: Time Features...")
    f['Tranc_Year'] = df['Tranc_Year'].astype(int)
    f['Tranc_Month'] = df['Tranc_Month'].astype(int)
    f['time_index'] = (df['Tranc_Year'] - 2012) * 12 + df['Tranc_Month']

    # ── Physical Features ───────────────────────────────────
    print ("build_features: Physical Features...", flush=True)
    logs.append("build_features: Physical Features...")
    f['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    f['mid_storey'] = pd.to_numeric(df['mid_storey'], errors='coerce')
    f['max_floor_lvl'] = pd.to_numeric(df['max_floor_lvl'], errors='coerce')
    f['hdb_age'] = pd.to_numeric(df['hdb_age'], errors='coerce')
    f['lease_commence_date']= pd.to_numeric(df['lease_commence_date'], errors='coerce')
    f['remaining_lease'] = (99-(df['Tranc_Year']-f['lease_commence_date'])).clip(0,99)
    f['total_dwelling_units'] = pd.to_numeric(df['total_dwelling_units'], errors='coerce')
    # Mapping for flat types (Section 4)
    flat_map = {'1 ROOM':1, 
                '2 ROOM':2, 
                '3 ROOM':3, 
                '4 ROOM':4, 
                '5 ROOM':5, 
                'EXECUTIVE':6, 
                'MULTI-GENERATION':7}
    f['flat_type_enc'] = df['flat_type'].map(flat_map).fillna(4)
    
    # ── Interaction Features ────────────────────────────────
    print ("build_features: Interaction Features...", flush=True)
    logs.append("build_features: Interaction Features...")
    f['floor_area_sqm_x_mid_storey'] = f['floor_area_sqm'] * f['mid_storey']
    f['lease_x_floor_area_sqm'] = (99 - f['hdb_age']) * f['floor_area_sqm']
    f['area_per_storey'] = f['floor_area_sqm'] / (f['mid_storey']+1)
    f['storey_ratio'] = f['mid_storey'] / (f['max_floor_lvl']+1)

    # ── MRT Features ─────────────────────────────────────────
    print ("build_features: MRT Features...", flush=True)
    logs.append("build_features: MRT Features...")
    f['mrt_nearest_distance'] = pd.to_numeric(df['mrt_nearest_distance'], errors='coerce')
    f['log_mrt_dist'] = np.log1p(f['mrt_nearest_distance'])
    f['mrt_interchange'] = pd.to_numeric(df['mrt_interchange'], errors='coerce').fillna(0)

    # ── Bus Features ─────────────────────────────────────────
    print ("build_features: Bus Features...", flush=True)
    logs.append("build_features: Bus Features...")
    f['bus_interchange'] = pd.to_numeric(df['bus_interchange'], errors='coerce').fillna(0)

    # ── Mall Features ────────────────────────────────────────
    print ("build_features: Mall Features...", flush=True)
    logs.append("build_features: Mall Features...")
    f['Mall_Nearest_Distance'] = pd.to_numeric(df['Mall_Nearest_Distance'], errors='coerce')
    f['log_mall_dist'] = np.log1p(f['Mall_Nearest_Distance'].fillna(664))
    f['Mall_Within_1km'] = pd.to_numeric(df['Mall_Within_1km'], errors='coerce').fillna(0)
    f['Mall_Within_2km'] = pd.to_numeric(df['Mall_Within_2km'], errors='coerce').fillna(0)

    # ── Hawker Features ───────────────────────────────────────
    print ("build_features: Hawker Features...", flush=True)
    logs.append("build_features: Hawker Features...")
    f['Hawker_Nearest_Distance'] = pd.to_numeric(df['Hawker_Nearest_Distance'], errors='coerce')
    f['log_hawker_dist'] = np.log1p(f['Hawker_Nearest_Distance'])
    f['Hawker_Within_1km'] = pd.to_numeric(df['Hawker_Within_1km'], errors='coerce').fillna(0)
    f['Hawker_Within_2km'] = pd.to_numeric(df['Hawker_Within_2km'], errors='coerce').fillna(0)
    f['hawker_food_stalls'] = pd.to_numeric(df['hawker_food_stalls'], errors='coerce')

    # ── School Features ───────────────────────────────────────
    print ("build_features: School Features...", flush=True)
    logs.append("build_features: School Features...")
    f['pri_sch_nearest_distance'] = pd.to_numeric(df['pri_sch_nearest_distance'], errors='coerce')
    f['pri_sch_affiliation'] = pd.to_numeric(df['pri_sch_affiliation'], errors='coerce').fillna(0)
    f['sec_sch_nearest_dist'] = pd.to_numeric(df['sec_sch_nearest_dist'], errors='coerce')
    f['cutoff_point'] = pd.to_numeric(df['cutoff_point'], errors='coerce').fillna(0)
    f['affiliation'] = pd.to_numeric(df['affiliation'], errors='coerce').fillna(0)

    # ── Geo Features ─────────────────────────────────────────
    print ("build_features: Geo Features...", flush=True)
    logs.append("build_features: Geo Features...")
    f['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    f['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # ── CBD Features ─────────────────────────────────────────
    print ("build_features: CBD Features...", flush=True)
    logs.append("build_features: CBD Features...")
    f['dist_cbd'] = np.sqrt((f['Latitude']-1.2837)**2+(f['Longitude']-103.8517)**2)*111

    # ── MSCP Features ────────────────────────────────────────
    print ("build_features: MSCP Features...", flush=True)
    logs.append("build_features: MSCP Features...")
    f['multistorey_carpark'] = (df['multistorey_carpark']=='Y').astype(int)
    
    # ── Postal Code Features ─────────────────────────────────
    print ("build_features: Postal Code Features...", flush=True)
    logs.append("build_features: Postal Code Features...")
    f['postal_code'] = pd.to_numeric(df['postal'], errors='coerce')
    
    # ── Categorical Features (Section 4/6) ──────────────────
    # CatBoost needs these as raw strings
    print ("build_features: Categorical Features...", flush=True)
    logs.append("build_features: Categorical Features...")
    CAT_COLS = ['town', 
                'flat_type', 
                'flat_model', 
                'planning_area', 
                'mrt_name',
                'full_flat_type',
                'address']
    for col in CAT_COLS:
        f[col] = df[col].astype(str)

    # Fill numeric NaN with median (Section 4)
    print ("build_features: Fill numeric Nan with median (Section 4)...", flush=True)
    logs.append("build_features: Fill numeric Nan with median (Section 4)...")
    num_cols = [c for c in f.columns if f[c].dtype != object]
    for col in num_cols:
        f[col] = f[col].fillna(f[col].median())

    # Final step in build_features(df):
    print ("build_features: Final step in build_features...", flush=True)
    logs.append("build_features: Final step in build_features...")
    MASTER_ORDER = ['Tranc_Year', 
                    'Tranc_Month', 
                    'time_index', 
                    'floor_area_sqm', 
                    'mid_storey',
                    'max_floor_lvl', 
                    'hdb_age', 
                    'lease_commence_date', 
                    'remaining_lease',
                    'total_dwelling_units', 
                    'flat_type_enc', 
                    'floor_area_sqm_x_mid_storey', 
                    'lease_x_floor_area_sqm',
                    'area_per_storey', 
                    'storey_ratio', 
                    'mrt_nearest_distance', 
                    'log_mrt_dist',
                    'mrt_interchange', 
                    'bus_interchange', 
                    'Mall_Nearest_Distance', 
                    'log_mall_dist',
                    'Mall_Within_1km', 
                    'Mall_Within_2km', 
                    'Hawker_Nearest_Distance', 
                    'log_hawker_dist',
                    'Hawker_Within_1km', 
                    'Hawker_Within_2km', 
                    'hawker_food_stalls', 
                    'pri_sch_nearest_distance',
                    'pri_sch_affiliation', 
                    'sec_sch_nearest_dist', 
                    'cutoff_point', 
                    'affiliation',
                    'Latitude', 
                    'Longitude', 
                    'dist_cbd', 
                    'multistorey_carpark', 
                    'postal_code', # Slot 37 filler
                    'town', 
                    'flat_type', 
                    'flat_model', 
                    'planning_area', 
                    'mrt_name', 
                    'address', 
                    'full_flat_type'
                   ]
    
    # Ensure the column names are strings of numbers for the model
    print ("build_features: Ensure the column names are strings of numbers for the model...", flush=True)
    logs.append("build_features: Ensure the column names are strings of numbers for the model...")
    f = f[MASTER_ORDER]
    f.columns = [str(i) for i in range(45)]

    status_report = {
        'messages': logs,
        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
    }
    
    return f, status_report