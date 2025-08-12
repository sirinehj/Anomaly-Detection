# Importing required libraries
import numpy as np
import pandas as pd
from yaml import CLoader as Loader, load

# Function to read the csv file
def read_data_csv(file_path, **kwargs):
    raw_data_csv = pd.read_csv(file_path  ,**kwargs)
    return raw_data_csv

# Function to find contamination score
def find_contamination(target_var,data):
    Fraud = data[data[target_var]==1]
    Valid = data[data[target_var]==0]
    contamination = len(Fraud)/float(len(Valid))
    print("Contamination: ",contamination)
    print("Fraud Class : {}".format(len(Fraud)))
    print("Normal Class : {}".format(len(Valid)))
    return contamination

# Function for reading config file
def read_config(path):
    with open(path) as stream:
        config = load(stream,Loader=Loader)
    return config
