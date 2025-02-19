import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Loads train, validation, and test CSV files from subdirectories in the input folder (`ML_data_crops`).
# Each dataset contains features and the target variable (`pm2.5`).
# Preprocesses the data by splitting into features (`X`) and target (`y`),
# and does z normalization using `StandardScaler`.
# Trains and tunes multiple regression models using GridSearchCV for hyperparameter optimization.
# Predicts `pm2.5` values for train, validation, and test sets and evaluates performance using MSE, MAE, RMSE, and R² metrics.
# Generates error analysis plots, including scatter plots, error histograms, etc.
# Each model folder contains:
# - A results file with performance metrics.
# - Plots visualizing prediction errors.

def load_data(train_path, validate_path, test_path):
    train_df = pd.read_csv(train_path)
    validate_df = pd.read_csv(validate_path)
    test_df = pd.read_csv(test_path)
    return train_df, validate_df, test_df

def preprocess_data(train_df, validate_df, test_df):
    X_train = train_df.drop(['pm2.5'], axis=1)
    y_train = train_df['pm2.5']

    X_validate = validate_df.drop(['pm2.5'], axis=1)
    y_validate = validate_df['pm2.5']

    X_test = test_df.drop(['pm2.5'], axis=1)
    y_test = test_df['pm2.5']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test, scaler

def scatter_plot_error(y_true, y_pred, output_path):
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(y_true, y_pred, c=errors, cmap='coolwarm', alpha=0.7, label="Predictions")
    plt.colorbar(scatter, label="Error Magnitude")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linewidth=2, label="Ideal Fit")
    plt.title("Scatter Plot with Error Magnitude")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def histogram_of_errors(errors, output_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True, color='blue')
    plt.title("Histogram of Prediction Errors")
    plt.xlabel("Error (Absolute Difference)")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()

def prediction_error_histogram(y_true, y_pred, output_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Zero Error")
    plt.title("Prediction Error Histogram")
    plt.xlabel("Prediction Error (Residuals)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def error_vs_ground_truth(y_true, y_pred, output_path):
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, errors, alpha=0.5)
    plt.title("Error vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Error Magnitude")
    plt.savefig(output_path)
    plt.close()

def create_scatter_plot(y_true, y_pred, output_path, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linewidth=2)
    plt.title(title)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(output_path)
    plt.close()

def get_hyperparameter_tuned_model(model_name, X_train, y_train, X_validate, y_validate):
    param_grids = {
        "LinearRegression": {},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "Lasso": {"alpha": [0.01, 0.1, 1.0]},
        "ElasticNet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        "RandomForestRegressor": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
        "GradientBoostingRegressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "AdaBoostRegressor": {"n_estimators": [50, 100, 200]},
        "ExtraTreesRegressor": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
        "SVR": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "epsilon": [0.01, 0.1, 1]},
        "KNeighborsRegressor": {"n_neighbors": [3, 5, 7]},
        "DecisionTreeRegressor": {"max_depth": [10, 20, None]},
        "GaussianProcessRegressor": {},
        "MLPRegressor": {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "learning_rate_init": [0.001, 0.01], "max_iter": [200, 500]}
    }

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=42),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "GaussianProcessRegressor": GaussianProcessRegressor(),
        "MLPRegressor": MLPRegressor(random_state=42)
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} is not supported.")

    print(f"Tuning hyperparameters for {model_name}...")
    grid_search = GridSearchCV(
        estimator=models[model_name],
        param_grid=param_grids[model_name],
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")

    y_validate_pred = best_model.predict(X_validate)
    mse = mean_squared_error(y_validate, y_validate_pred)
    r2 = r2_score(y_validate, y_validate_pred)
    print(f"Validation MSE: {mse:.2f}, R²: {r2:.2f}")

    return best_model

def process_triplet(input_folder, output_root_folder):
    train_path = os.path.join(input_folder, "train_data.csv")
    validate_path = os.path.join(input_folder, "validation_data.csv")
    test_path = os.path.join(input_folder, "test_data.csv")

    subfolder_name = os.path.basename(input_folder)
    output_subfolder = os.path.join(output_root_folder, subfolder_name)
    os.makedirs(output_subfolder, exist_ok=True)

    train_df, validate_df, test_df = load_data(train_path, validate_path, test_path)

    X_train, y_train, X_validate, y_validate, X_test, y_test, scaler = preprocess_data(train_df, validate_df, test_df)

    model_names = [
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForestRegressor",
        "GradientBoostingRegressor", "AdaBoostRegressor", "ExtraTreesRegressor",
        "SVR", "KNeighborsRegressor", "DecisionTreeRegressor",
        "GaussianProcessRegressor", "MLPRegressor"
    ]

    for model_name in model_names:
        print(f"Processing model: {model_name} for subfolder: {subfolder_name}...")
        model_folder = os.path.join(output_subfolder, model_name)
        os.makedirs(model_folder, exist_ok=True)

        try:
            best_model = get_hyperparameter_tuned_model(model_name, X_train, y_train, X_validate, y_validate)

            y_train_pred = best_model.predict(X_train)
            y_validate_pred = best_model.predict(X_validate)
            y_test_pred = best_model.predict(X_test)
            errors = np.abs(y_test - y_test_pred)

            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_train_pred)

            validate_mse = mean_squared_error(y_validate, y_validate_pred)
            validate_mae = mean_absolute_error(y_validate, y_validate_pred)
            validate_rmse = np.sqrt(validate_mse)
            validate_r2 = r2_score(y_validate, y_validate_pred)

            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, y_test_pred)

            result = (
                f"\nModel: {model_name}\n"
                f"Training Set: MSE = {train_mse:.2f}, MAE = {train_mae:.2f}, RMSE = {train_rmse:.2f}, R² = {train_r2:.2f}\n"
                f"Validation Set: MSE = {validate_mse:.2f}, MAE = {validate_mae:.2f}, RMSE = {validate_rmse:.2f}, R² = {validate_r2:.2f}\n"
                f"Test Set: MSE = {test_mse:.2f}, MAE = {test_mae:.2f}, RMSE = {test_rmse:.2f}, R² = {test_r2:.2f}\n"
            )
            result_file_path = os.path.join(model_folder, f"{model_name}_results.txt")
            with open(result_file_path, "w") as f:
                f.write(result)

            scatter_error_path = os.path.join(model_folder, f"{model_name}_scatter_error.png")
            histogram_of_errors_path = os.path.join(model_folder, f"{model_name}_histogram_of_errors.png")
            prediction_error_histogram_path = os.path.join(model_folder, f"{model_name}_prediction_error_histogram.png")
            error_vs_ground_truth_path = os.path.join(model_folder, f"{model_name}_error_vs_ground_truth.png")
            scatter_basic_path = os.path.join(model_folder, f"{model_name}_scatter_basic.png")

            scatter_plot_error(y_test, y_test_pred, scatter_error_path)
            histogram_of_errors(errors, histogram_of_errors_path)
            prediction_error_histogram(y_test, y_test_pred, prediction_error_histogram_path)
            error_vs_ground_truth(y_test, y_test_pred, error_vs_ground_truth_path)
            create_scatter_plot(y_test, y_test_pred, scatter_basic_path, f"{model_name} Scatter Plot")

            print(f"Results and plots saved for {model_name} in {subfolder_name}.")

        except Exception as e:
            error_message = f"Model {model_name} failed with error: {e}\n"
            print(error_message)
            error_file_path = os.path.join(model_folder, f"{model_name}_error.txt")
            with open(error_file_path, "w") as f:
                f.write(error_message)

def main():
    input_root_folder = "ML_data_crops"
    output_root_folder = "ML_all_models"

    os.makedirs(output_root_folder, exist_ok=True)

    for subfolder in os.listdir(input_root_folder):
        subfolder_path = os.path.join(input_root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            process_triplet(subfolder_path, output_root_folder)

    print("Processing completed. Results saved in the 'results' folder.")

if __name__ == "__main__":
    main()
