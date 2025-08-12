# Importing required libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from numpy import quantile, where, random
import pickle

# Function to train the model with better parameters
def train_IF(data, contamination=None):
    """
    Train Isolation Forest with improved parameters
    """
    if contamination is None:
        contamination = 0.0018  # Default value
    
    # Try different parameter combinations
    clf = IsolationForest(
        n_estimators=500, 
        max_samples='auto',  # Let it choose automatically
        contamination=contamination,
        random_state=42,     # For reproducible results
        n_jobs=-1           # Use all CPU cores
    )
    clf.fit(data)
    return clf

# Alternative training function with different approach
def train_IF_alternative(data):
    """
    Alternative Isolation Forest with different strategy
    """
    clf = IsolationForest(
        n_estimators=1000,   # More trees
        max_samples=0.8,     # Use 80% of data for each tree
        contamination=0.001, # Lower contamination rate
        random_state=42,
        bootstrap=False,     # Don't bootstrap samples
        n_jobs=-1
    )
    clf.fit(data)
    return clf

# Function to predict scores
def predict_scores(model, data):
    scores_prediction = model.decision_function(data)
    return scores_prediction

# Function to get predictions with custom threshold
def predict_with_threshold(model, data, threshold_percentile=99.8):
    """
    Make predictions using a custom threshold instead of contamination parameter
    """
    scores = model.decision_function(data)
    threshold = quantile(scores, threshold_percentile/100.0)
    predictions = where(scores < threshold, -1, 1)
    return predictions

# Function to train lof model with better parameters
def train_lof(data, contamination=None, n_neighbors=None):
    if contamination is None:
        contamination = 0.0018
    if n_neighbors is None:
        n_neighbors = 20
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,  # Set to True if you want to predict on new data
        n_jobs=-1
    )
    lof.fit_predict(data)
    return lof

# Function to calculate anomaly score
def anomaly_scores(model):
    anomaly_scores = model.negative_outlier_factor_ 
    return anomaly_scores

# Function to save model
def save_model(model, framework, model_path):
    if framework == "IF":
        model_path += '/IF_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        model_path += '/LOF_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    print('model saved at: ', model_path)
    return model

# Function to tune contamination parameter
def find_best_contamination(data, target, contamination_values=None):
    """
    Test different contamination values to find the best one
    """
    if contamination_values is None:
        contamination_values = [0.0001, 0.0005, 0.001, 0.0018, 0.003, 0.005, 0.01]
    
    best_contamination = None
    best_recall = 0
    results = {}
    
    for cont in contamination_values:
        clf = IsolationForest(
            n_estimators=100,  # Fewer estimators for faster tuning
            contamination=cont,
            random_state=42
        )
        clf.fit(data)
        predictions = clf.predict(data)
        
        # Calculate recall (how many frauds we caught)
        y_pred = (predictions == -1).astype(int)
        tp = ((target == 1) & (y_pred == 1)).sum()  # True positives
        fn = ((target == 1) & (y_pred == 0)).sum()  # False negatives
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results[cont] = recall
        
        if recall > best_recall:
            best_recall = recall
            best_contamination = cont
    
    print("Contamination tuning results:")
    for cont, recall in results.items():
        print(f"Contamination {cont}: Recall = {recall:.3f}")
    
    print(f"Best contamination: {best_contamination} (Recall: {best_recall:.3f})")
    return best_contamination, results