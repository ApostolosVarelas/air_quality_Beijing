import pandas as pd

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

lag_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']

for col in lag_cols:
    for i in range(1, 11):
        data[f'{col}_lag{i}'] = data[col].shift(i)

lag_columns = [f'{col}_lag{i}' for col in lag_cols for i in range(1, 3)]
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

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(validate_data)}")
print(f"Test size: {len(test_data)}")

train_data.to_csv('train_data.csv', index=False)
validate_data.to_csv('validate_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
