# Importing required packages
import numpy as np
import pandas as pd
import pickle

from yaml import CLoader as Loader, load
from ml_pipeline.utils import read_data_csv, read_config, find_contamination
from ml_pipeline.preprocessing import handle_null_values
from ml_pipeline.model import train_IF,train_lof, predict_scores, anomaly_scores, save_model


# Reading config file
config = read_config("modular_code/input/config.yaml")

# Reading the data
transaction_data = read_data_csv(config['data_path'])

# Handling missing values
transaction_data = handle_null_values(transaction_data)

# Calculate contamination score
contamination_score = find_contamination('Class',transaction_data)

# Dropping the target variable
X = transaction_data.drop('Class',axis=1)
y = transaction_data['Class']

# Training the isolation forest model
clf = train_IF(X)
print("Isolation forest model trained successfully")

# Predicting isolation forest scores
scores_prediction = predict_scores(clf,X)

transaction_data['scores'] = scores_prediction

# Saving isolation forest model
save_model(clf,"IF",config['model_path'])

# Training the LOF model
lof = train_lof(X)
print("LOF model trained successfully")

# Finding anomaly score
anomaly_scores = anomaly_scores(lof)
print("Anomaly scores for the LOF model:-",anomaly_scores)

# Saving LOF model
save_model(lof,"LOF",config['model_path'])