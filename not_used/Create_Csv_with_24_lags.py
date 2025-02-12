import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Loads air quality data from a CSV file
# Handles missing values using forward-fill (`ffill`) and backward-fill (`bfill`).
# Creates lagged features for `DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir` with time lags up to 24 hours.
# Drops any remaining missing values after feature engineering.

# Splits the dataset into:
# - Training data (years before 2014).
# - Validation data (70% of 2014 data).
# - Test data (remaining 30% of 2014 data).

# Saves processed train, validation, and test datasets as CSV files

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def feature_engineering(df):
    df = pd.get_dummies(df, columns=['cbwd'], drop_first=True)

    df.drop(['No'], axis=1, inplace=True)
    return df

def handle_missing_values(df):
    return df.bfill().ffill()

def split_data(df):
    train_data = df[df['year'] < 2014]
    validation_data = df[df['year'] == 2014].iloc[:int(len(df[df['year'] == 2014]) * 0.7)]
    test_data = df[df['year'] == 2014].iloc[int(len(df[df['year'] == 2014]) * 0.7):]
    return train_data, validation_data, test_data

def normalize_data(train, validation, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    validation_scaled = scaler.transform(validation)
    test_scaled = scaler.transform(test)
    return train_scaled, validation_scaled, test_scaled, scaler

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, mae, r2

def create_lagged_features(df, features, max_lag):
    lagged_data = {}
    for feature in features:
        for lag in range(1, max_lag + 1):
            lagged_data[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    lagged_df = pd.DataFrame(lagged_data)
    df = pd.concat([df, lagged_df], axis=1)
    
    df = df.dropna()

    return df

def main():
    file_path = "PRSA_data_2010.1.1-2014.12.31.csv" 
    df = load_data(file_path)

    df = handle_missing_values(df)

    df = feature_engineering(df)

    features_to_lag = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    max_lag = 24 
    df = create_lagged_features(df, features_to_lag, max_lag)

    df = df.dropna()

    train_data, validation_data, test_data = split_data(df)

    train_data.to_csv("train_data.csv", index=False)
    validation_data.to_csv("validation_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)

    print("SIUU")

if __name__ == "__main__":
    main()