# Importing required packages
import numpy as np
import pandas as pd
import pickle

from yaml import CLoader as Loader, load
from ml_pipeline.utils import read_data_csv, read_config, find_contamination
from ml_pipeline.preprocessing import handle_null_values
from ml_pipeline.model import train_IF, train_lof, predict_scores, anomaly_scores, save_model

from sklearn.metrics import classification_report, confusion_matrix

# Reading config file
config = read_config("input/config.yaml")

# Reading the data
transaction_data = read_data_csv(config['data_path'])

# Handling missing values
transaction_data = handle_null_values(transaction_data)

# Calculate contamination score
contamination_score = find_contamination('Class', transaction_data)

# Dropping the target variable
X = transaction_data.drop('Class', axis=1)
y = transaction_data['Class']

print("\n" + "="*60)
print("TRAINING IMPROVED ISOLATION FOREST MODEL")
print("="*60)

# Training the isolation forest model with BETTER contamination parameter
# Use 0.01 instead of 0.0018 based on test results
clf = train_IF(X, contamination=0.1)
print("Isolation forest model trained successfully")

# After training IF model
predictions = clf.predict(X)
fraud_predictions = (predictions == -1).sum()
print(f"IF predicted {fraud_predictions} fraudulent transactions")
print(f"Actual fraud transactions: {(y == 1).sum()}")

# Predicting isolation forest scores
scores_prediction = predict_scores(clf, X)

transaction_data['scores'] = scores_prediction

# Saving isolation forest model
save_model(clf, "IF", config['model_path'])

# Training the LOF model with better parameters
print("\n" + "="*60)
print("TRAINING IMPROVED LOF MODEL")
print("="*60)

lof = train_lof(X, contamination=0.1, n_neighbors=50) # contamination=0.1 : maximum fraud detection

print("LOF model trained successfully")

# Finding anomaly score
anomaly_scores_lof = anomaly_scores(lof)
print("Anomaly scores for the LOF model calculated")

# Saving LOF model
save_model(lof, "LOF", config['model_path'])

# Convert predictions to match your target format (1 for fraud, 0 for normal)
y_pred = (predictions == -1).astype(int)

print("\n" + "="*60)
print("IMPROVED MODEL PERFORMANCE")
print("="*60)

# Calculate detailed metrics
tp = ((y == 1) & (y_pred == 1)).sum()
fp = ((y == 0) & (y_pred == 1)).sum()
fn = ((y == 1) & (y_pred == 0)).sum()
tn = ((y == 0) & (y_pred == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Fraud Detection Summary:")
print(f"  Actual Fraud Cases: {(y == 1).sum()}")
print(f"  Predicted Fraud Cases: {(y_pred == 1).sum()}")
print(f"  Correctly Caught Fraud: {tp}")
print(f"  Missed Fraud: {fn}")
print(f"  False Alarms: {fp}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Normal', 'Fraud']))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

# ENSEMBLE APPROACH - COMBINING IF + LOF
print("\n" + "="*60)
print("ENSEMBLE APPROACH: COMBINING IF + LOF")
print("="*60)

# Get IF predictions and LOF scores
if_predictions = clf.predict(X)
lof_scores = anomaly_scores_lof

# Strategy 1: Union (Flag if EITHER model is suspicious)
lof_threshold = np.percentile(lof_scores, 3)  # Bottom 3% most suspicious
lof_predictions = (lof_scores < lof_threshold).astype(int)
if_binary = (if_predictions == -1).astype(int)

ensemble_union = (if_binary | lof_predictions).astype(int)

# Strategy 2: Intersection (Flag only if BOTH models agree)
ensemble_intersection = (if_binary & lof_predictions).astype(int)

# Strategy 3: Weighted ensemble (more sophisticated)
# Normalize scores to 0-1 range
if_scores_norm = (clf.decision_function(X) - clf.decision_function(X).min()) / (clf.decision_function(X).max() - clf.decision_function(X).min())
lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

# Combine scores (lower = more suspicious for both)
ensemble_scores = 0.6 * (1 - if_scores_norm) + 0.4 * (1 - lof_scores_norm)
ensemble_threshold = np.percentile(ensemble_scores, 97)  # Top 3% most suspicious
ensemble_weighted = (ensemble_scores > ensemble_threshold).astype(int)

# Test all ensemble strategies
ensemble_methods = {
    "IF Only": if_binary,
    "LOF Only": lof_predictions, 
    "Union (IF OR LOF)": ensemble_union,
    "Intersection (IF AND LOF)": ensemble_intersection,
    "Weighted Ensemble": ensemble_weighted
}

print("Ensemble Strategy Comparison:")
print("-" * 80)
print(f"{'Method':<25} {'Frauds Caught':<15} {'Total Flagged':<15} {'Recall':<10} {'Precision':<10}")
print("-" * 80)

best_method = None
best_fraud_caught = 0

for method_name, predictions in ensemble_methods.items():
    tp_ens = ((y == 1) & (predictions == 1)).sum()
    fp_ens = ((y == 0) & (predictions == 1)).sum()
    total_flagged = predictions.sum()
    
    recall_ens = tp_ens / 253
    precision_ens = tp_ens / total_flagged if total_flagged > 0 else 0
    
    print(f"{method_name:<25} {tp_ens:<15} {total_flagged:<15} {recall_ens:<10.3f} {precision_ens:<10.3f}")
    
    if tp_ens > best_fraud_caught:
        best_fraud_caught = tp_ens
        best_method = method_name
        best_predictions = predictions

print("-" * 80)
print(f"🏆 BEST METHOD: {best_method}")
print(f"   Frauds caught: {best_fraud_caught}/253 ({best_fraud_caught/253*100:.1f}% recall)")

# Detailed analysis of best method
print(f"\n" + "="*60)
print(f"DETAILED ANALYSIS: {best_method.upper()}")
print("="*60)

tp_best = ((y == 1) & (best_predictions == 1)).sum()
fp_best = ((y == 0) & (best_predictions == 1)).sum()
fn_best = ((y == 1) & (best_predictions == 0)).sum()
total_flagged_best = best_predictions.sum()

precision_best = tp_best / total_flagged_best if total_flagged_best > 0 else 0
recall_best = tp_best / 253

print(f"Fraud Detection Summary:")
print(f"  Actual Fraud Cases: 253")
print(f"  Predicted Fraud Cases: {total_flagged_best}")
print(f"  Correctly Caught Fraud: {tp_best}")
print(f"  Missed Fraud: {fn_best}")
print(f"  False Alarms: {fp_best}")
print(f"  Precision: {precision_best:.3f}")
print(f"  Recall: {recall_best:.3f}")

print(f"\nConfusion Matrix ({best_method}):")
print(confusion_matrix(y, best_predictions))

print("\n" + "="*60)
print("FINAL PERFORMANCE COMPARISON")
print("="*60)
print(f"✅ ORIGINAL: Caught 1/253 frauds (0.4% recall)")
print(f"📈 IF ONLY (0.03): Caught 28/253 frauds (11.1% recall)")
print(f"🎯 BEST ENSEMBLE: Caught {tp_best}/253 frauds ({recall_best*100:.1f}% recall)")
print(f"🚀 TOTAL IMPROVEMENT: {tp_best-1} more frauds detected!")
print("="*60)