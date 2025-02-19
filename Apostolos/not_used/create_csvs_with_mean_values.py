import pandas as pd
import numpy as np

"""
This code create 3 csv files from the original dataset:
train_data
validation_data
test_data

Those csvs contains all the features plus:
For each feature it has the mean value
For each feature it has the last 2 lags

We tested it but it was not usefull enough  
"""

file_path = 'PRSA_data_2010.1.1-2014.12.31.csv'
data = pd.read_csv(file_path)

data['cbwd'] = data['cbwd'].astype('category').cat.codes

data_daily = (
    data
    .groupby(['year', 'month', 'day'], as_index=False)
    .agg({
        'pm2.5': 'mean',
        'DEWP': 'mean',
        'TEMP': 'mean',
        'PRES': 'mean',
        'cbwd': 'mean',  
        'Iws': 'mean',
        'Is': 'mean',
        'Ir': 'mean'
    })
)

data = data_daily

data['pm2.5_change'] = data['pm2.5'].diff().apply(
    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
)

data = data[data['pm2.5_change'] != 0]

lag_cols = ['TEMP', 'PRES', 'Is', 'Ir']

for col in lag_cols:
    for i in range(1, 3):
        data[f'{col}_lag{i}'] = data[col].shift(i)

min_dewp = data['DEWP'].min()
max_dewp = data['DEWP'].max()
mean_dewp = data['DEWP'].mean()

data['DEWP_scaled'] = ((data['DEWP'] - mean_dewp) / (max_dewp - min_dewp)) * 20

data['DEWP_scaled'] = data['DEWP_scaled'] - data['DEWP_scaled'].mean()

data['DEWP_scaled_lag0'] = data['DEWP_scaled']
data['DEWP_scaled_lag1'] = data['DEWP_scaled'].shift(1)
data['DEWP_scaled_lag2'] = data['DEWP_scaled'].shift(2)

min_cbwd = data['cbwd'].min()
max_cbwd = data['cbwd'].max()
mean_cbwd = data['cbwd'].mean()

min_iws = data['Iws'].min()
max_iws = data['Iws'].max()
mean_iws = data['Iws'].mean()

data['cbwd_scaled'] = ((data['cbwd'] - mean_cbwd) / (max_cbwd - min_cbwd)) * 20
data['cbwd_scaled'] = data['cbwd_scaled'] - data['cbwd_scaled'].mean()

data['Iws_scaled'] = ((data['Iws'] - mean_iws) / (max_iws - min_iws)) * 20
data['Iws_scaled'] = data['Iws_scaled'] - data['Iws_scaled'].mean()

data['cbwd_scaled_lag0'] = data['cbwd_scaled']  
data['cbwd_scaled_lag1'] = data['cbwd_scaled'].shift(1)  
data['cbwd_scaled_lag2'] = data['cbwd_scaled'].shift(2) 

data['Iws_scaled_lag0'] = data['Iws_scaled'] 
data['Iws_scaled_lag1'] = data['Iws_scaled'].shift(1)  
data['Iws_scaled_lag2'] = data['Iws_scaled'].shift(2)  

lag_columns = [f'{col}_lag{i}' for col in lag_cols for i in range(1, 3)]
lag_columns.extend(['DEWP_scaled_lag0', 'DEWP_scaled_lag1', 'DEWP_scaled_lag2'])
lag_columns.extend([
    'cbwd_scaled_lag0', 'cbwd_scaled_lag1', 'cbwd_scaled_lag2',
    'Iws_scaled_lag0', 'Iws_scaled_lag1', 'Iws_scaled_lag2'
])

data.dropna(subset=lag_columns, inplace=True)

train_data = data[data['year'] < 2014]

validate_data = data[
    (data['year'] == 2014) & 
    ((data['month'] < 7) | ((data['month'] == 6) & (data['day'] <= 30)))
]

test_data = data[
    (data['year'] == 2014) & 
    ((data['month'] > 6) | ((data['month'] == 7) & (data['day'] >= 1)))
]

train_data.drop(columns=['year'], inplace=True)
validate_data.drop(columns=['year'], inplace=True)
test_data.drop(columns=['year'], inplace=True)

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(validate_data)}")
print(f"Test size: {len(test_data)}")

train_data.to_csv('train_data.csv', index=False)
validate_data.to_csv('validation_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
