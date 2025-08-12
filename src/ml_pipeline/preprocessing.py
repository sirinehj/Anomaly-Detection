# Function for Null values imputation
def handle_null_values(data):
    data = data.fillna(data.median())
    return data

