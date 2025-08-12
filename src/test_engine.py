# Importing required packages
import numpy as np
import pandas as pd
import pickle

from yaml import CLoader as Loader, load
from ml_pipeline.utils import read_data_csv, read_config, find_contamination
from ml_pipeline.preprocessing import handle_null_values
from ml_pipeline.model import (train_IF, train_lof, predict_scores, anomaly_scores, save_model,
                              find_best_contamination, train_IF_alternative, predict_with_threshold)

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Helper function to evaluate and print model performance"""
    print(f"\n{'='*50}")
    print(f"{model_name} Results:")
    print(f"{'='*50}")
    
    # Basic counts
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Fraud Detection Summary:")
    print(f"  Actual Fraud Cases: {(y_true == 1).sum()}")
    print(f"  Predicted Fraud Cases: {(y_pred == 1).sum()}")
    print(f"  Correctly Caught Fraud: {tp}")
    print(f"  Missed Fraud: {fn}")
    print(f"  False Alarms: {fp}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return precision, recall

# Reading config file
config = read_config("input/config.yaml")

# Reading the data
transaction_data = read_data_csv(config['data_path'])
print(f"Dataset loaded: {transaction_data.shape}")

# Handling missing values
transaction_data = handle_null_values(transaction_data)

# Calculate contamination score
contamination_score = find_contamination('Class', transaction_data)

# Dropping the target variable
X = transaction_data.drop('Class', axis=1)
y = transaction_data['Class']

print(f"\nStarting model testing and tuning...")

# Test 1: Find best contamination parameter
print("\n" + "="*60)
print("TEST 1: FINDING BEST CONTAMINATION PARAMETER")
print("="*60)

best_contamination, tuning_results = find_best_contamination(X, y)

# Test 2: Train with best contamination
print(f"\n" + "="*60)
print("TEST 2: TRAINING WITH BEST CONTAMINATION")
print("="*60)

clf_best = train_IF(X, contamination=best_contamination)
predictions_best = clf_best.predict(X)
y_pred_best = (predictions_best == -1).astype(int)
evaluate_model(y, y_pred_best, "Isolation Forest (Best Contamination)")

# Test 3: Alternative approach
print(f"\n" + "="*60)
print("TEST 3: ALTERNATIVE ISOLATION FOREST")
print("="*60)

clf_alt = train_IF_alternative(X)
predictions_alt = clf_alt.predict(X)
y_pred_alt = (predictions_alt == -1).astype(int)
evaluate_model(y, y_pred_alt, "Alternative Isolation Forest")

# Test 4: Custom threshold approach
print(f"\n" + "="*60)
print("TEST 4: CUSTOM THRESHOLD APPROACH")
print("="*60)

# Train a model and use custom threshold
clf_threshold = train_IF(X, contamination=0.1)  # Higher contamination for more flexibility
predictions_threshold = predict_with_threshold(clf_threshold, X, threshold_percentile=99.8)
y_pred_threshold = (predictions_threshold == -1).astype(int)
evaluate_model(y, y_pred_threshold, "Custom Threshold")

# Test different threshold percentiles
print(f"\nTesting different thresholds:")
for percentile in [99.5, 99.6, 99.7, 99.8, 99.9]:
    pred_thresh = predict_with_threshold(clf_threshold, X, threshold_percentile=percentile)
    y_pred_thresh = (pred_thresh == -1).astype(int)
    tp = ((y == 1) & (y_pred_thresh == 1)).sum()
    fp = ((y == 0) & (y_pred_thresh == 1)).sum()
    recall = tp / (y == 1).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  Threshold {percentile}%: Recall={recall:.3f}, Precision={precision:.3f}, Flagged={y_pred_thresh.sum()}")

print(f"\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
print("Choose the approach that gives you the best balance of precision and recall!")