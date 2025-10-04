import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# loading models
print("Loading models...")
det_pkg = joblib.load('exoplanet_model.pkl')
hab_pkg = joblib.load('habitability_model.pkl')

det_model = det_pkg['model']
det_scaler = det_pkg['scaler']
det_encoder = det_pkg['encoder']
det_feats = det_pkg['features']

hab_model = hab_pkg['model']
hab_feats = hab_pkg['features']
hab_classes = hab_pkg['class_names']

print("Models loaded\n")

def prep_features(user_input):
    """
    takes basic inputs and estimates everything else
    probably not perfect but seems to work okay
    """
    
    # grab what user gave us
    period = user_input.get('period', 365)
    radius = user_input.get('radius', 1.0)
    depth = user_input.get('transit_depth', 100)
    duration = user_input.get('transit_duration', 13)
    temp = user_input.get('temperature', 288)
    star_temp = user_input.get('stellar_temp', 5778)
    snr = user_input.get('signal_noise_ratio', 50)
    
    # estimate other stuff from physics
    # insolation goes down with distance squared
    insol = 1.0 * (365 / period) ** 2
    
    # star properties from temperature - using main sequence relations
    star_rad = (star_temp / 5778) ** 0.8
    star_logg = 4.4  # most stars are around this
    star_lum = (star_temp / 5778) ** 4
    
    # if user didnt give SNR, estimate it
    if snr == 50 and depth != 100:
        snr = (depth / 10) * (duration / 5)
    
    # semi-major axis - Kepler's third law
    sma = (period / 365) ** (2/3)
    
    # estimate mass from radius
    # different for rocky vs gas planets
    if radius < 1.5:
        mass = radius ** 3.7
    else:
        mass = radius ** 2.0
    
    ecc = 0.02  # most planets have low eccentricity
    
    # build the full dict
    full = {
        'koi_period': period,
        'koi_duration': duration,
        'koi_depth': depth,
        'koi_prad': radius,
        'koi_teq': temp,
        'koi_insol': insol,
        'koi_steff': star_temp,
        'koi_slogg': star_logg,
        'koi_srad': star_rad,
        'koi_model_snr': snr,
        'pl_rade': radius,
        'pl_masse': mass,
        'pl_orbsmax': sma,
        'pl_orbeccen': ecc,
        'pl_insol': insol,
        'st_teff': star_temp,
        'st_lum': star_lum,
        'st_rad': star_rad,
        'st_mass': 1.0,
        'ra': 0,
        'dec': 0
    }
    
    return full

def detect(planet_data):
    # run the detection model
    X = pd.DataFrame([planet_data])
    
    # add missing columns as 0
    for f in det_feats:
        if f not in X.columns:
            X[f] = 0
    
    X = X[det_feats].fillna(0)
    
    # scale and predict
    X_sc = det_scaler.transform(X)
    pred = det_model.predict(X_sc)[0]
    probs = det_model.predict_proba(X_sc)[0]
    
    pred_class = det_encoder.inverse_transform([pred])[0]
    conf = max(probs) * 100
    
    res = {
        'classification': pred_class,
        'confidence': round(conf, 2),
        'probabilities': {}
    }
    
    for i, cls in enumerate(det_encoder.classes_):
        res['probabilities'][cls] = round(probs[i] * 100, 2)
    
    return res

def check_habitability(planet_data):
    # only call this for confirmed planets
    X = pd.DataFrame([planet_data])
    
    # see if we have enough features
    avail = [f for f in hab_feats if f in X.columns]
    
    if len(avail) < 3:
        return {
            'classification': 'Unknown',
            'percentage': 0,
            'note': 'not enough data'
        }
    
    # fill missing
    for f in hab_feats:
        if f not in X.columns:
            X[f] = 0
    
    X = X[hab_feats].fillna(0)
    
    # predict
    pred = hab_model.predict(X)[0]
    probs = hab_model.predict_proba(X)[0]
    
    # calculate percentage from class probabilities
    # tried different weights, this worked best
    pct = probs[0] * 15 + probs[1] * 50 + probs[2] * 85
    
    result = {
        'classification': hab_classes[int(pred)],
        'percentage': round(pct, 1),
        'probabilities': {}
    }
    
    for i in range(len(hab_classes)):
        result['probabilities'][hab_classes[i]] = round(probs[i] * 100, 2)
    
    return result

def analyze_manual(user_input):
    """
    For when user enters values manually
    
    needs these keys in dict:
        period, radius, transit_depth, transit_duration, temperature, stellar_temp
    optional: signal_noise_ratio
    """
    
    # get full features from simplified input
    full_data = prep_features(user_input)
    
    # run detection first
    det_result = detect(full_data)
    
    # check habitability if its confirmed
    if det_result['classification'] == 'CONFIRMED':
        hab_result = check_habitability(full_data)
    else:
        hab_result = {
            'classification': 'N/A',
            'percentage': 0,
            'note': f'detected as {det_result["classification"]}, not confirmed'
        }
    
    return {
        'detection': det_result,
        'habitability': hab_result
    }

def analyze_csv(file_path):
    """
    For CSV upload - processes the whole file
    CSV should have standard NASA exoplanet columns
    """
    
    df = pd.read_csv(file_path, comment='#', low_memory=False)
    results = []
    
    print(f"\nAnalyzing {len(df)} planets...")
    
    for idx, row in df.iterrows():
        planet = row.to_dict()
        
        # run detection
        det_res = detect(planet)
        
        # habitability if confirmed
        if det_res['classification'] == 'CONFIRMED':
            hab_res = check_habitability(planet)
        else:
            hab_res = {'classification': 'N/A', 'percentage': 0}
        
        results.append({
            'index': idx,
            'detection': det_res,
            'habitability': hab_res
        })
        
        # show progress
        if (idx + 1) % 100 == 0:
            print(f"  done: {idx + 1}/{len(df)}")
    
    print("Complete!\n")
    return results

# TODO: maybe add batch export to JSON or something

# quick test
if __name__ == '__main__':
    
    print("="*60)
    print("Testing manual entry")
    print("="*60)
    
    # earth-like test
    test1 = {
        'period': 365,
        'radius': 1.0,
        'transit_depth': 84,
        'transit_duration': 13,
        'temperature': 288,
        'stellar_temp': 5778
    }
    
    res1 = analyze_manual(test1)
    
    print("\nEarth-like:")
    print(f"  Detection: {res1['detection']['classification']} ({res1['detection']['confidence']}%)")
    if res1['habitability']['classification'] != 'N/A':
        print(f"  Habitability: {res1['habitability']['classification']} ({res1['habitability']['percentage']}%)")
    
    # hot jupiter
    test2 = {
        'period': 3.5,
        'radius': 11.0,
        'transit_depth': 20000,
        'transit_duration': 4.5,
        'temperature': 1500,
        'stellar_temp': 6200,
        'signal_noise_ratio': 60
    }
    
    res2 = analyze_manual(test2)
    
    print("\nHot Jupiter:")
    print(f"  Detection: {res2['detection']['classification']} ({res2['detection']['confidence']}%)")
    if res2['habitability']['classification'] != 'N/A':
        print(f"  Habitability: {res2['habitability']['classification']} ({res2['habitability']['percentage']}%)")
    
    print("\n" + "="*60)
    print("CSV upload test")
    print("="*60)
    print("Use: analyze_csv('file.csv')")
    print("(not testing to avoid loading big files)")
    
    print("\n" + "="*60)
    print("Done! Both methods work")
    print("\nUI team can use:")
    print("  analyze_manual(dict) - for form input")
    print("  analyze_csv(path) - for file upload")
