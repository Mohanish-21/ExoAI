import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def calculate_esi(radius, stellar_flux):
    # Earth Similarity Index - closer to 1 means more Earth-like
    radius_sim = 1 - abs(radius - 1) / (radius + 1)
    flux_sim = 1 - abs(stellar_flux - 1) / (stellar_flux + 1)
    esi = (radius_sim * flux_sim) ** 0.5
    return max(0, min(1, esi))

def calculate_hzd(semi_major_axis, stellar_lum):
    # Habitable Zone Distance calc from Kasting et al
    hz_inner = 0.75 * np.sqrt(stellar_lum)
    hz_outer = 1.77 * np.sqrt(stellar_lum)
    
    if hz_outer == hz_inner:
        return 0
    
    hzd = (2 * semi_major_axis - hz_outer - hz_inner) / (hz_outer - hz_inner)
    return hzd

def calculate_equilibrium_temp(stellar_temp, stellar_lum, semi_major, albedo=0.3):
    # equilibrium temp in Kelvin - fixed the syntax issue
    if semi_major == 0:
        return 0
    temp = stellar_temp * ((stellar_lum / (4 * semi_major**2))**0.25) * ((1 - albedo)**0.25)
    return temp

def hab_score(esi, hzd, temp):
    # tried different weight combos, these gave best results
    weights = {'esi': 0.4, 'hzd': 0.3, 'temp': 0.3}
    
    # ESI scoring
    esi_score = 100 if esi >= 0.8 else max(0, (esi / 0.8) * 100)
    
    # HZD scoring - sweet spot is -1 to 1
    hzd_score = 100 if -1 <= hzd <= 1 else max(0, 100 - 50 * abs(hzd))
    
    # temp scoring for liquid water (260-320K ideal)
    if 260 <= temp <= 320:
        temp_score = 100
    elif temp < 260:
        temp_score = max(0, 100 - 2 * (260 - temp))
    else:
        temp_score = max(0, 100 - 2 * (temp - 320))
    
    percent = (weights['esi'] * esi_score + weights['hzd'] * hzd_score + weights['temp'] * temp_score)
    return min(100, max(0, percent))

def classify_hab(percent):
    if percent >= 70:
        return 2  # highly habitable
    elif percent >= 30:
        return 1  # potentially habitable
    else:
        return 0  # not habitable

if __name__ == '__main__':
    print("Loading datasets for habitability analysis...")
    
    # load all three missions - they should have good mix of features
    try:
        kepler = pd.read_csv('your_kepler_file.csv', comment='#', low_memory=False)
        kepler['mission'] = 'Kepler'
        print(f"Kepler: {len(kepler)} rows, {len(kepler.columns)} cols")
    except:
        kepler = None
        print("Kepler file not found")
    
    try:
        k2 = pd.read_csv('your_k2_file.csv', comment='#', low_memory=False)
        k2['mission'] = 'K2'
        print(f"K2: {len(k2)} rows, {len(k2.columns)} cols")
    except:
        k2 = None
        print("K2 file not found")
    
    try:
        tess = pd.read_csv('your_tess_file.csv', comment='#', low_memory=False)
        tess['mission'] = 'TESS'
        print(f"TESS: {len(tess)} rows, {len(tess.columns)} cols")
    except:
        tess = None
        print("TESS file not found")
    
    # combine available datasets
    datasets = [d for d in [kepler, k2, tess] if d is not None]
    if not datasets:
        print("No datasets loaded!")
        exit()
    
    df = pd.concat(datasets, ignore_index=True, sort=False)
    print(f"\nCombined: {len(df)} total exoplanets")
    
    # check which habitability features we actually have
    hab_features = {
        'radius': ['pl_rade', 'koi_prad'],  # planet radius
        'flux': ['pl_insol', 'koi_insol'],  # insolation flux
        'sma': ['pl_orbsmax', 'koi_sma'],  # semi-major axis
        'lum': ['st_lum'],  # stellar luminosity (might not exist)
        'temp': ['st_teff', 'koi_steff']  # stellar temp
    }
    
    # find which columns exist
    col_map = {}
    for feat, options in hab_features.items():
        for opt in options:
            if opt in df.columns:
                col_map[feat] = opt
                break
    
    print(f"\nFound features: {col_map}")
    
    # need at least radius, flux, and temp for basic calc
    if 'radius' not in col_map or 'flux' not in col_map:
        print("Error: Missing critical features for habitability")
        exit()
    
    # calculate ESI (always possible with radius and flux)
    print("\nCalculating Earth Similarity Index...")
    df['esi'] = df.apply(lambda row: calculate_esi(
        row[col_map['radius']] if pd.notna(row[col_map['radius']]) else 1.0,
        row[col_map['flux']] if pd.notna(row[col_map['flux']]) else 1.0
    ), axis=1)
    
    # calculate HZD if we have sma and luminosity
    if 'sma' in col_map and 'lum' in col_map:
        print("Calculating Habitable Zone Distance...")
        df['hzd'] = df.apply(lambda row: calculate_hzd(
            row[col_map['sma']] if pd.notna(row[col_map['sma']]) else 1.0,
            row[col_map['lum']] if pd.notna(row[col_map['lum']]) else 1.0
        ), axis=1)
    else:
        # fake it with approximate values based on flux
        print("Warning: Missing sma or luminosity, approximating HZD from flux")
        df['hzd'] = df[col_map['flux']].apply(lambda x: 0 if 0.5 < x < 1.5 else 1.5)
    
    # calculate equilibrium temp
    if 'temp' in col_map and 'lum' in col_map and 'sma' in col_map:
        print("Calculating equilibrium temperatures...")
        df['eq_temp'] = df.apply(lambda row: calculate_equilibrium_temp(
            row[col_map['temp']] if pd.notna(row[col_map['temp']]) else 5778,
            row[col_map['lum']] if pd.notna(row[col_map['lum']]) else 1.0,
            row[col_map['sma']] if pd.notna(row[col_map['sma']]) else 1.0
        ), axis=1)
    else:
        # use existing temp if available
        if 'koi_teq' in df.columns:
            df['eq_temp'] = df['koi_teq']
        elif 'pl_eqt' in df.columns:
            df['eq_temp'] = df['pl_eqt']
        else:
            print("Warning: Can't calculate temperature, using defaults")
            df['eq_temp'] = 288  # earth-like default
    
    # calculate habitability percentage
    print("Calculating habitability scores...")
    df['hab_percent'] = df.apply(lambda row: hab_score(
        row['esi'], row['hzd'], row['eq_temp']
    ), axis=1)
    
    df['hab_class'] = df['hab_percent'].apply(classify_hab)
    
    print(f"\nHabitability distribution:")
    print(df['hab_class'].value_counts())
    print(f"\nAverage habitability: {df['hab_percent'].mean():.1f}%")
    
    # prepare ML features
    ml_features = ['pl_rade', 'pl_masse', 'st_teff', 'st_lum', 'pl_orbsmax', 
                   'pl_orbeccen', 'pl_insol', 'koi_prad', 'koi_period', 'koi_teq']
    
    available = [f for f in ml_features if f in df.columns]
    print(f"\nUsing {len(available)} features for model: {available}")
    
    X = df[available].fillna(0)
    y = df['hab_class']
    
    # remove nulls
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]
    
    print(f"Training on {len(X)} planets")
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # train model - tried 50, 100, 150 trees
    # 100 was good balance of speed and accuracy
    print("\nTraining model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"Accuracy: {acc*100:.2f}%")
    
    # save model
    hab_pkg = {
        'model': model,
        'features': available,
        'accuracy': acc,
        'class_names': {0: 'Not Habitable', 1: 'Potentially Habitable', 2: 'Highly Habitable'}
    }
    
    joblib.dump(hab_pkg, 'habitability_model.pkl')
    print("\nModel saved as habitability_model.pkl")
    
    # test predictions
    print("\nSample predictions:")
    for i in range(min(5, len(X_test))):
        idx = np.random.randint(0, len(X_test))
        sample = X_test.iloc[idx:idx+1]
        true_class = int(y_test.iloc[idx])
        pred_class = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        
        # calc hab percent from probs
        hab_pct = proba[0] * 15 + proba[1] * 50 + proba[2] * 85
        
        status = "correct" if true_class == pred_class else "wrong"
        print(f"  True: {hab_pkg['class_names'][true_class]:25s} | "
              f"Pred: {hab_pkg['class_names'][int(pred_class)]:25s} | "
              f"Hab: {hab_pct:.1f}% ({status})")
    
    print("\nDone!")
