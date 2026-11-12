import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

RAW_FILE = "wafer_sensor_fault_detection.csv"

def extract_batch_timestamp(filename):
    """
    Extracts timestamp from Batch_File like 'wafer_07012020_041011.csv'
    to order wafers chronologically for drift detection.
    """
    if not isinstance(filename, str):
        return pd.NaT
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        date_str, time_str = match.groups()
        # Try DDMMYYYY first, fallback to MMDDYYYY
        for fmt in ("%d%m%Y_%H%M%S", "%m%d%Y_%H%M%S"):
            try:
                return datetime.strptime(f"{date_str}_{time_str}", fmt)
            except ValueError:
                continue
    return pd.NaT

def clean_and_reduce_wafer_data(filepath):
    print(f"1. Loading {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"   Raw Shape: {df.shape}")

    # Separate metadata and sensor columns
    meta_cols = ['Wafer', 'Good/Bad', 'Source', 'Batch_File']
    sensor_cols = [c for c in df.columns if c.startswith('Sensor-')]

    # 2. Extract Batch Timestamp for temporal drift tracking
    df['Batch_Timestamp'] = df['Batch_File'].apply(extract_batch_timestamp)

    # 3. Handle Known Corrupted String Columns & Coerce to Float
    # Sensor-332 is a calibration timestamp log -> drop it from numeric sensors
    if 'Sensor-332' in sensor_cols:
        print("   Dropping Sensor-332 (calibration date string column)...")
        sensor_cols.remove('Sensor-332')
        df.drop(columns=['Sensor-332'], inplace=True)

    # Convert 'invalid' (Sensor-576), 'string ' (Sensor-586), and any other artifacts to NaN
    print("   Coercing sensor artifacts ('invalid', 'string ') to numeric NaN...")
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Drop Sensors with Excessive Missing Data (> 50% NaN)
    null_ratios = df[sensor_cols].isna().mean()
    high_null_cols = null_ratios[null_ratios > 0.50].index.tolist()
    print(f"   Dropping {len(high_null_cols)} sensors with greater than 50% missing values...")
    sensor_cols = [c for c in sensor_cols if c not in high_null_cols]
    df.drop(columns=high_null_cols, inplace=True)

    # 5. Drop Zero-Variance Sensors (like Sensor-6 = 100.0)
    # Fit ONLY on the labeled training pool to prevent data leakage
    train_mask = df['Good/Bad'].notna()
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(df.loc[train_mask, sensor_cols]),
        columns=sensor_cols,
        index=df.loc[train_mask].index
    )

    selector = VarianceThreshold(threshold=1e-5)
    selector.fit(X_train_imputed)
    kept_variance_cols = [col for col, keep in zip(sensor_cols, selector.get_support()) if keep]
    dropped_var_count = len(sensor_cols) - len(kept_variance_cols)
    print(f"   Dropping {dropped_var_count} zero-variance/constant sensors (e.g., locked setpoints)...")
    
    sensor_cols = kept_variance_cols
    X_train_imputed = X_train_imputed[sensor_cols]

    # 6. Drop Highly Collinear Sensors (Pearson r > 0.95)
    corr_matrix = X_train_imputed.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    print(f"   Dropping {len(collinear_cols)} highly collinear redundant sensors (r > 0.95)...")
    
    final_sensor_cols = [c for c in sensor_cols if c not in collinear_cols]
    print(f"   Final Reduced Sensor Count: {len(final_sensor_cols)} active inline sensors")

    # 7. Impute the Full Dataset using the Training Median Imputer
    final_imputer = SimpleImputer(strategy='median')
    final_imputer.fit(df.loc[train_mask, final_sensor_cols])
    
    df_sensors_clean = pd.DataFrame(
        final_imputer.transform(df[final_sensor_cols]),
        columns=final_sensor_cols,
        index=df.index
    )

    # Combine metadata and cleaned sensors
    clean_df = pd.concat([df[meta_cols + ['Batch_Timestamp']], df_sensors_clean], axis=1)

    # Map Target for XGBoost compatibility: -1 (Defective) -> 1 (Fault/Drift), +1 (Normal) -> 0 (Normal)
    clean_df['Is_Defective'] = clean_df['Good/Bad'].map({-1.0: 1, 1.0: 0})

    # Split into Labeled Pool and Unlabeled Production Stream
    labeled_df = clean_df[clean_df['Good/Bad'].notna()].copy()
    unlabeled_df = clean_df[clean_df['Good/Bad'].isna()].copy()

    # Sort unlabeled stream chronologically by Batch_Timestamp for drift visualization
    unlabeled_df.sort_values(by=['Batch_Timestamp', 'Wafer'], inplace=True)

    labeled_df.to_csv("clean_labeled_wafers.csv", index=False)
    unlabeled_df.to_csv("clean_unlabeled_stream.csv", index=False)

    print("\n--- Summary ---")
    print(f"Saved Labeled Training Set: {labeled_df.shape} -> 'clean_labeled_wafers.csv'")
    print(f"Saved Unlabeled Production Stream: {unlabeled_df.shape} -> 'clean_unlabeled_stream.csv'")

if __name__ == "__main__":
    clean_and_reduce_wafer_data(RAW_FILE)