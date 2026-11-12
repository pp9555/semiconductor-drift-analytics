import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, f1_score
)
from xgboost import XGBClassifier

# 1. Load your cleaned datasets from Step 1
print("1. Loading cleaned datasets...")
labeled_df = pd.read_csv("clean_labeled_wafers.csv")
unlabeled_df = pd.read_csv("clean_unlabeled_stream.csv")

sensor_cols = [c for c in labeled_df.columns if c.startswith('Sensor-')]
X = labeled_df[sensor_cols]

# Target: 1 = Defective (-1 in raw), 0 = Normal (+1 in raw)
y = labeled_df['Is_Defective'].astype(int)

# Stratified Split (keeping 25% held-out test set just like the Kaggle notebook to compare!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Narrow 257 Sensors -> Top 30 using Mutual Information (Trained ONLY on X_train)
print("2. Ranking sensors via Mutual Information to beat the Curse of Dimensionality...")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=sensor_cols).sort_values(ascending=False)

top_30_sensors = mi_series.head(30).index.tolist()
print(f"   Top 5 Informational Sensors: {top_30_sensors[:5]}")

# 3. Train Regulated XGBoost with Imbalance Weighting
# Because Normal (0) is the tiny class (67 samples) and Defective (1) is 1085 samples,
# scale_pos_weight = count(0) / count(1) down-weights the overwhelming Defective class!
neg_count = (y_train == 0).sum() # Normal wafers (67)
pos_count = (y_train == 1).sum() # Defective wafers (1085)
balance_ratio = neg_count / pos_count

print(f"3. Training XGBoost (Imbalance ratio weight = {balance_ratio:.4f})...")
xgb_model = XGBClassifier(
    n_estimators=120,
    max_depth=3,              # Shallow trees prevent memorizing noise on 67 normal samples
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=balance_ratio,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train[top_30_sensors], y_train)

# 4. Evaluate on Held-Out Test Set & Compare with the Failed Kaggle Notebook
y_prob_defective = xgb_model.predict_proba(X_test[top_30_sensors])[:, 1]
y_prob_normal = 1.0 - y_prob_defective

# Optimize threshold using balanced F1 on training predictions
train_probs = xgb_model.predict_proba(X_train[top_30_sensors])[:, 1]
best_thresh = 0.50
best_macro_f1 = 0.0
for thresh in np.arange(0.20, 0.85, 0.02):
    score = f1_score(y_train, (train_probs >= thresh).astype(int), average='macro')
    if score > best_macro_f1:
        best_macro_f1 = score
        best_thresh = thresh

y_pred = (y_prob_defective >= best_thresh).astype(int)

print("\n================ BENCHMARK VS KAGGLE NOTEBOOK ================")
print(f"Optimal Decision Threshold: {best_thresh:.2f}")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Defective (1)']))
print(f"ROC-AUC Score:                     {roc_auc_score(y_test, y_prob_defective):.4f}  (Kaggle Notebook was 0.6190)")
print(f"PR-AUC (Normal Class Detection):   {average_precision_score(1 - y_test, y_prob_normal):.4f}  (Kaggle Notebook was 0.0958)")
print("==============================================================\n")

# 5. Extract Top 10 Root-Cause Sensors for the Micron Drift Dashboard
importances = pd.Series(xgb_model.feature_importances_, index=top_30_sensors).sort_values(ascending=False)
top_10_drift_sensors = importances.head(10).index.tolist()
print("4. Top 10 Root-Cause Sensors driving excursions:")
for rank, (sensor, imp) in enumerate(importances.head(10).items(), 1):
    print(f"   #{rank}: {sensor} (Importance: {imp:.4f})")

# 6. Build the "Golden Wafer Baseline" & Calculate EWMA Drift on Unlabeled Production Stream
print("\n5. Computing Golden Baseline & EWMA Drift on the 1,833 Unlabeled Wafers...")
# In a real fab, control limits (UCL/LCL) are set from the mean & std of KNOWN GOOD wafers
golden_wafers = labeled_df[labeled_df['Is_Defective'] == 0]

baseline_stats = {}
for sensor in top_10_drift_sensors:
    mean_val = float(golden_wafers[sensor].mean())
    std_val = float(golden_wafers[sensor].std())
    if std_val < 1e-6:
        std_val = float(labeled_df[sensor].std()) or 1.0
    baseline_stats[sensor] = {
        "golden_mean": mean_val,
        "golden_std": std_val,
        "ucl_3sigma": mean_val + 3 * std_val,
        "lcl_3sigma": mean_val - 3 * std_val
    }

# Score the Unlabeled Production Stream with XGBoost + EWMA Drift Metrics
unlabeled_df['Defect_Probability'] = xgb_model.predict_proba(unlabeled_df[top_30_sensors])[:, 1]
unlabeled_df['Predicted_Status'] = np.where(
    unlabeled_df['Defect_Probability'] >= best_thresh, 'Excursion_Alert', 'In_Control'
)

# For each of the Top 5 sensors, compute Z-score shift from Golden Baseline and EWMA (lambda=0.2)
for sensor in top_10_drift_sensors[:5]:
    g_mean = baseline_stats[sensor]['golden_mean']
    g_std = baseline_stats[sensor]['golden_std']
    
    # Z-score measures how many standard deviations the wafer is from a Golden Wafer
    z_col = f"{sensor}_Z_Score"
    ewma_col = f"{sensor}_EWMA"
    
    unlabeled_df[z_col] = ((unlabeled_df[sensor] - g_mean) / g_std).clip(-10, 10)
    # EWMA smooths noise across sequential wafers to expose gradual tool drift
    unlabeled_df[ewma_col] = unlabeled_df[sensor].ewm(alpha=0.2, adjust=False).mean()

# Save artifacts for our FastAPI + Next.js Dashboard
dashboard_cols = [
    'Wafer', 'Batch_File', 'Batch_Timestamp', 'Defect_Probability', 'Predicted_Status'
] + top_10_drift_sensors + [f"{s}_Z_Score" for s in top_10_drift_sensors[:5]] + [f"{s}_EWMA" for s in top_10_drift_sensors[:5]]

unlabeled_df[dashboard_cols].to_csv("scored_production_drift_stream.csv", index=False)

with open("golden_baseline_metadata.json", "w") as f:
    json.dump({
        "top_10_sensors": top_10_drift_sensors,
        "feature_importance": importances.head(10).to_dict(),
        "optimal_threshold": float(best_thresh),
        "baseline_limits": baseline_stats
    }, f, indent=2)

print("   Saved 'scored_production_drift_stream.csv' and 'golden_baseline_metadata.json'!")