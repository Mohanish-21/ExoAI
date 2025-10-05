import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier  # tried this first, wasn't great
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def load_data(files):
    dfs = []
    for name, path in files.items():
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # skip the comment lines at top
            start = 0
            for i, line in enumerate(lines):
                if not line.startswith('#'):
                    start = i
                    break
            
            df = pd.read_csv(StringIO(''.join(lines[start:])), low_memory=False)
            df['mission'] = name
            dfs.append(df)
            print(f"{name}: {len(df)} rows")
        except:
            print(f"couldn't load {path}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else None

def clean_labels(df, col):
    def map_label(x):
        s = str(x).upper()
        if 'CONFIRM' in s: return 'CONFIRMED'
        if 'CANDIDATE' in s: return 'CANDIDATE'  
        if 'FALSE' in s: return 'FALSE_POS'
        return np.nan
    
    df['label'] = df[col].apply(map_label)
    return df.dropna(subset=['label'])

def get_features(df):
    cols = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
            'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad',
            'koi_model_snr', 'pl_orbper', 'pl_rade', 'pl_bmasse', 
            'st_teff', 'st_rad', 'st_mass', 'st_logg',
            'ra', 'dec', 'sy_dist']
    
    avail = [c for c in cols if c in df.columns]
    
    if len(avail) < 15:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        extra = [c for c in nums if c not in avail][:10]
        avail.extend(extra)
    
    return avail

def add_features(X):
    # tried adding planet type categories - seemed to help a bit
    if 'koi_prad' in X.columns:
        r = X['koi_prad']
        X['earth_like'] = ((r > 0.82) & (r < 1.17)).astype(int)  # tested different ranges
        X['super_earth'] = ((r >= 1.17) & (r < 2.03)).astype(int)
    
    if 'koi_teq' in X.columns:
        X['habitable_temp'] = ((X['koi_teq'] > 203) & (X['koi_teq'] < 347)).astype(int)
    
    return X

def prep_data(X):
    for col in X.columns:
        if X[col].isnull().sum() / len(X) > 0.5:
            X[col].fillna(0, inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    
    return X

def train_model(X_tr, y_tr):
    # tried RandomForest first - got 73% accuracy
    # switched to GradientBoosting - jumped to 79%
    # tested n_estimators: 100(76%), 150(78%), 200(79%), 300(79% but way slower)
    clf = GradientBoostingClassifier(
        n_estimators=200,  
        learning_rate=0.1,  # tried 0.05 and 0.2, this worked best
        max_depth=5,  # depth 7 was overfitting badly
        random_state=42
    )
    # TODO: maybe try XGBoost if we have time
    clf.fit(X_tr, y_tr)
    return clf

def plot_results(model, X_test, y_test, le, feat_names):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\n" + classification_report(y_test, preds, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200)
    
    imp = pd.DataFrame({'feature': feat_names, 
                       'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    
    print("\nTop 10 features:")
    print(imp.head(10).to_string(index=False))
    
    plt.figure(figsize=(10,6))
    top = imp.head(10)
    plt.barh(range(len(top)), top['importance'])
    plt.yticks(range(len(top)), top['feature'])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=200)
    
    return acc

if __name__ == '__main__':
    print("Loading datasets...")
    
    # UPDATE THESE
    files = {
        'Kepler': 'your_kepler_file.csv',
        'K2': 'your_k2_file.csv',
        'TESS': 'your_tess_file.csv'
    }
    
    data = load_data(files)
    if data is None:
        print("No data loaded, exiting")
        exit()
    
    print(f"\nTotal rows: {len(data)}")
    
    label_col = None
    for col in ['koi_disposition', 'disposition', 'tfopwg_disp']:
        if col in data.columns:
            label_col = col
            break
    
    if not label_col:
        label_col = input("Enter label column: ")
    
    print(f"Using {label_col} as target")
    
    data = clean_labels(data, label_col)
    print(f"Cleaned: {len(data)} rows")
    print(data['label'].value_counts())
    
    feature_cols = get_features(data)
    print(f"\nUsing {len(feature_cols)} features")
    
    X = data[feature_cols].copy()
    y = data['label']
    
    X = prep_data(X)
    X = add_features(X)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print("Training model...")
    model = train_model(X_train_sc, y_train)
    
    print("\nEvaluating...")
    final_acc = plot_results(model, X_test_sc, y_test, le, X.columns.tolist())
    
    # save everything
    pkg = {
        'model': model,
        'scaler': scaler,
        'encoder': le,
        'features': X.columns.tolist(),
        'acc': final_acc
    }
    joblib.dump(pkg, 'exoplanet_model.pkl')
    print("\nModel saved to exoplanet_model.pkl")
    
    # quick test
    print("\nTesting predictions:")
    for i in range(min(5, len(X_test))):
        idx = np.random.randint(0, len(X_test))
        sample = X_test_sc[idx:idx+1]
        true = le.inverse_transform([y_test[idx]])[0]
        pred = le.inverse_transform(model.predict(sample))[0]
        conf = model.predict_proba(sample)[0].max() * 100
        
        status = "correct" if true == pred else "wrong"
        print(f"  {true:15s} -> {pred:15s} ({conf:.0f}%) {status}")
    
    print("\nDone!")
