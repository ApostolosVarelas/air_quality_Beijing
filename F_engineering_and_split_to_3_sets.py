import pandas as pd
import numpy as np

# Loads air quality data from CSV file.
# Handles missing values by applying forward-fill (`ffill`) and backward-fill (`bfill`).

# Extracts date information and splits data into train (before 2013), validation (2013-01-01 to 2013-08-31), and test sets (from 2013-09-01 onward).

# Performs feature engineering:
# - Converts hour, month, and day into sin and cos too.
# - Encodes wind direction (`cbwd`) into numerical categories.
# - Drops unnecessary columns (`No`, `year`).

# Creates lagged features for `DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir`, shifting values by 1 to 4 time steps. This has been changed depending on the circumstance. Different feature engineering has been applied for the creation of different csv triples.
# Removes the `date` column after processing.
# Saves processed train, validation, and test datasets as CSV files


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    return df.bfill().ffill()

def feature_engineering(df):
    df = df.copy()
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    wind_directions = {'NW': 0, 'NE': 1, 'SE': 2, 'SW': 3}
    df['cbwd'] = df['cbwd'].map(wind_directions)
    
    df.drop(['No', 'year'], axis=1, inplace=True)
    
    return df

def create_lagged_features(df, features):
    df = df.copy()
    for feature in features:
        df.loc[:, f'{feature}_lag_1'] = df[feature].shift(1)
        df.loc[:, f'{feature}_lag_2'] = df[feature].shift(2)
        df.loc[:, f'{feature}_lag_3'] = df[feature].shift(3)
        df.loc[:, f'{feature}_lag_4'] = df[feature].shift(4)
    
    df.dropna(inplace=True)
    return df

def main():
    file_path = "PRSA_data_2010.1.1-2014.12.31.csv"
    
    df = load_data(file_path)
    
    df = handle_missing_values(df)
    
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    train_df = df[df['date'] < '2013-01-01']
    validation_df = df[(df['date'] >= '2013-01-01') & (df['date'] < '2013-09-01')]
    test_df = df[df['date'] >= '2013-09-01']
    
    train_df = feature_engineering(train_df)
    validation_df = feature_engineering(validation_df)
    test_df = feature_engineering(test_df)
    
    features_to_lag = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    
    train_df = create_lagged_features(train_df, features_to_lag)
    validation_df = create_lagged_features(validation_df, features_to_lag)
    test_df = create_lagged_features(test_df, features_to_lag)
    
    train_df.drop(['date'], axis=1, inplace=True)
    validation_df.drop(['date'], axis=1, inplace=True)
    test_df.drop(['date'], axis=1, inplace=True)
    
    train_df.to_csv("train_data.csv", index=False)
    validation_df.to_csv("validation_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    
    print("DONE")

if __name__ == "__main__":
    main()


