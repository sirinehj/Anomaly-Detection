# Importing required libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from numpy import quantile, where, random
import pickle

# Function to train the model
def train_IF(data):
    clf=IsolationForest(n_estimators=500, max_samples=len(data),contamination=0.0018)
    clf.fit(data)
    return clf

# Function to predict scores
def predict_scores(model, data):
    scores_prediction = model.decision_function(data)
    return scores_prediction

# Function to train lof model
def train_lof(data):
    lof = LocalOutlierFactor(n_neighbors=20,contamination=0.0018)
    lof.fit_predict(data)
    return lof

# Function to calculate anomaly score
def anomaly_scores(model):
    anomaly_scores = model.negative_outlier_factor_ 
    return anomaly_scores

# Function to save model
def save_model(model,framework,model_path):
    if framework=="IF":
        model_path += '/IF_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        model_path += '/LOF_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    print('model saved at: ', model_path)
    return model