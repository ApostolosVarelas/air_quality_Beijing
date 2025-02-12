import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

train_data = pd.read_csv('train_data.csv')
train_data = train_data.dropna()
validate_data = pd.read_csv('validate_data.csv')
test_data = pd.read_csv('test_data.csv')

# ------------------ FEATURE & TARGET SELECTION ---------------------
X_train = train_data.drop(columns=['pm2.5_change', 'year', 'month', 'day', 'pm2.5'])
y_train = train_data['pm2.5_change']

X_validate = validate_data.drop(columns=['pm2.5_change', 'year', 'month', 'day', 'pm2.5'])
y_validate = validate_data['pm2.5_change']

X_test = test_data.drop(columns=['pm2.5_change', 'year', 'month', 'day', 'pm2.5'])
y_test = test_data['pm2.5_change']

# ------------------ STANDARDIZATION -------------------------------
feature_names = train_data.drop(columns=['pm2.5_change', 'year', 'month', 'day', 'pm2.5']).columns

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_validate = pd.DataFrame(scaler.transform(X_validate), columns=feature_names)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)


# ------------------ CLASSIFIERS & PARAM GRID -----------------------
classifiers = {
    'RandomForest': (
        RandomForestClassifier(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(), {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        }
    ),
    'LogisticRegression': (
        LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    ),
    'DecisionTree': (
        DecisionTreeClassifier(), {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    ),
    'KNeighbors': (
        KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    ),
    'SVM': (
        SVC(probability=True), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    ),
}

f1_scores = {'Classifier': [], 'Set': [], 'F1-Score': []}
# ------------------ GRID SEARCH & EVALUATION -----------------------
for name, (clf, param_grid) in classifiers.items():
    output_text = []
    output_text.append(f"\nClassifier: {name}")
    
    print(f"\nClassifier: {name}")
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("  Best Parameters:", grid_search.best_params_)
    output_text.append(f"  Best Parameters: {grid_search.best_params_}")
    
    y_val_pred = grid_search.predict(X_validate)
    print("  Validation Report:")
    print(classification_report(y_validate, y_val_pred))
    output_text.append("  Validation Report:")
    output_text.append(classification_report(y_validate, y_val_pred))
    
    val_f1_weighted = f1_score(y_validate, y_val_pred, average='weighted')
    print(f"  Weighted F1-Score (Validation): {val_f1_weighted:.4f}")
    output_text.append(f"  Weighted F1-Score (Validation): {val_f1_weighted:.4f}")
    f1_scores['Classifier'].append(name)
    f1_scores['Set'].append('Validation')
    f1_scores['F1-Score'].append(val_f1_weighted)
    
    cm_val = confusion_matrix(y_validate, y_val_pred)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=np.unique(y_train))
    disp_val.plot(cmap='Blues')
    plt.title(f'{name} - Confusion Matrix (Validation)')
    plt.savefig(f'{name}_confusion_matrix_validation.png')
    plt.close()
    
    y_test_pred = grid_search.predict(X_test)
    print("  Test Report:")
    print(classification_report(y_test, y_test_pred))
    output_text.append("  Test Report:")
    output_text.append(classification_report(y_test, y_test_pred))
    
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    print(f"  Weighted F1-Score (Test): {test_f1_weighted:.4f}")
    output_text.append(f"  Weighted F1-Score (Test): {test_f1_weighted:.4f}")
    f1_scores['Classifier'].append(name)
    f1_scores['Set'].append('Test')
    f1_scores['F1-Score'].append(test_f1_weighted)
    
    cm_test = confusion_matrix(y_test, y_test_pred)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.unique(y_train))
    disp_test.plot(cmap='Blues')
    plt.title(f'{name} - Confusion Matrix (Test)')
    plt.savefig(f'{name}_confusion_matrix_test.png')
    plt.close()
    
    if hasattr(grid_search.best_estimator_, 'feature_importances_'):
        importances = grid_search.best_estimator_.feature_importances_
        features = X_train.columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.title(f'{name} - Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.savefig(f'{name}_feature_importances.png')
        plt.close()

    if hasattr(grid_search.best_estimator_, "predict_proba") or hasattr(grid_search.best_estimator_, "decision_function"):
        plt.figure(figsize=(8, 6))
        if hasattr(grid_search.best_estimator_, "predict_proba"):
            y_val_prob = grid_search.predict_proba(X_validate)[:, 1]
        else:
            y_val_prob = grid_search.decision_function(X_validate)

        fpr, tpr, _ = roc_curve(y_validate, y_val_prob, pos_label=np.unique(y_validate)[-1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f'{name}_roc_curve.png')
        plt.close()
    
    output_file_path = "classification_results.txt"

    with open(output_file_path, "a+") as file:
        file.write("\n".join(output_text))

print("F1-Score Comparison Plot")
f1_df = pd.DataFrame(f1_scores)
plt.figure(figsize=(12, 6))
sns.barplot(data=f1_df, x='Classifier', y='F1-Score', hue='Set')
plt.title('F1-Score Comparison Across Classifiers')
plt.ylabel('Weighted F1-Score')
plt.xlabel('Classifier')
plt.xticks(rotation=45)
plt.legend(title='Dataset')
plt.tight_layout()
plt.savefig('f1_score_comparison.png')
plt.close()
