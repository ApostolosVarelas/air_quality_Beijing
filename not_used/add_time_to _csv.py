import numpy as np
import pandas as pd

"""
This code used to edit the already created csv file, to add time circular time feutures
"""

# Function to add time engineering features
def add_time_features(data):
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    return data

# Load datasets
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
validation_data = pd.read_csv('validation_data.csv')

# Apply time engineering
train_data = add_time_features(train_data)
test_data = add_time_features(test_data)
validation_data = add_time_features(validation_data)

# Save the modified datasets if needed
train_data.to_csv('train_data_with_time_features.csv', index=False)
test_data.to_csv('test_data_with_time_features.csv', index=False)
validation_data.to_csv('validation_data_with_time_features.csv', index=False)
