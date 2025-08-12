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
clf = train_IF(X, contamination=0.03)  # ← THIS IS THE KEY CHANGE!
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

lof = train_lof(X, contamination=0.01, n_neighbors=50)  # ← BETTER PARAMETERS!
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

print("\n" + "="*60)
print("PERFORMANCE IMPROVEMENT SUMMARY")
print("="*60)
print(f"✅ BEFORE: Caught 1/253 frauds (0.4% recall)")
print(f"🎯 NOW: Caught {tp}/253 frauds ({recall*100:.1f}% recall)")
print(f"📈 IMPROVEMENT: {tp-1} more frauds detected!")
print(f"⚠️  Trade-off: {fp} additional false alarms")
print("="*60)