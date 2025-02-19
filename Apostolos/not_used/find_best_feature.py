import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

"""
This code creates 3 CSV files from the original dataset:
train_data
validation_data
test_data

These CSVs contain all the features with the optimal lag number. Meaning that when we say 'TEMP': 21, we mean from lag 1 until lag 21.

The optimal lag numbers can be found by performing cross-validation for each feature separately. 
Specifically, for each feature, we test multiple lag depths (from 1 to max_lag) and evaluate their performance using a RandomForestRegressor model. 
The performance is measured using root mean squared error (RMSE) through 5-fold cross-validation. 
The lag depth that results in the lowest mean RMSE is selected as the optimal lag for that feature.

We tested it, but it was not useful enough.
"""

def find_optimal_lag_depth_cv(data, target, features, max_lag):
    """
    Determine the optimal lag depth for features using cross-validation.

    Args:
    data (pd.DataFrame): Dataset with lagged features.
    target (str): Target column name.
    features (list): List of feature names (excluding lags).
    max_lag (int): Maximum lag depth to evaluate.

    Returns:
    dict: Optimal lags for each feature with RMSE scores.
    """
    results = {}
    X = data.drop(columns=[target])
    y = data[target]
    
    for feature in features:
        rmse_scores = []
        for lag in range(1, max_lag + 1):
            lagged_feature = f'{feature}_lag{lag}'
            if lagged_feature in data.columns:
                X_subset = X[[lagged_feature]]
                
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                model = RandomForestRegressor(random_state=42)
                
                scorer = make_scorer(mean_squared_error, squared=False)
                
                cv_scores = cross_val_score(model, X_subset, y, cv=kf, scoring=scorer)
                mean_rmse = np.mean(cv_scores)
                rmse_scores.append((lag, mean_rmse))
        
        optimal_lag = min(rmse_scores, key=lambda x: x[1])
        results[feature] = optimal_lag
    
    return results

def add_lagged_features(df, features, max_lag):
    """
    Add lagged features to a DataFrame up to the specified maximum lag for each feature.

    Args:
    df (pd.DataFrame): Original DataFrame.
    features (list): List of feature names to generate lags for.
    max_lag (int): Maximum lag depth to create.

    Returns:
    pd.DataFrame: DataFrame with lagged features added.
    """
    df_with_lags = df.copy()
    
    for feature in features:
        for lag in range(1, max_lag + 1):
            df_with_lags[f'{feature}_lag{lag}'] = df_with_lags[feature].shift(lag)
    
    return df_with_lags

train_data_path = 'final_feature_engineering/train_data_with_lags.csv'
validation_data_path = 'final_feature_engineering/validation_data_with_lags.csv'
test_data_path = 'final_feature_engineering/test_data_with_lags.csv'

train_data = pd.read_csv(train_data_path)
validation_data = pd.read_csv(validation_data_path)
test_data = pd.read_csv(test_data_path)

target_column = 'pm2.5'
original_features = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']  
max_lag_depth = 24

optimal_lags = find_optimal_lag_depth_cv(train_data, target_column, original_features, max_lag_depth)

print("Optimal lag depths for each feature:")
print(optimal_lags)

selected_lag_features = [f"{feature}_lag{optimal_lags[feature][0]}" for feature in optimal_lags]
X_train = train_data[selected_lag_features]
y_train = train_data[target_column]

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

X_val = validation_data[selected_lag_features]
y_val = validation_data[target_column]

X_val = X_val.dropna()
y_val = y_val.loc[X_val.index]

model = RandomForestRegressor(
    random_state=42,
    max_depth=10,          
    min_samples_split=10,   
    min_samples_leaf=5,   
    n_estimators=100        
)
model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

print(f"Validation RMSE: {val_rmse}")

X_test = test_data[selected_lag_features]
y_test = test_data[target_column]

X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

test_predictions = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"Test RMSE: {test_rmse}")
