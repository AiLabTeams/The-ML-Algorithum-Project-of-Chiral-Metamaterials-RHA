
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor


# List of Excel files to load
file_paths = ['Au_Ag_RHA_peak_value_and parameters.xlsx']

# Define the columns and y_column
x_columns = ['Oad1', 'Inc1', 'dOad', 'dInc', 'Oad2', 'Inc2']
y_column = 'CD_abs'

# Create empty lists to store various importance values and scores
r2_scores_lightgbm = []
rmse_scores_lightgbm = []
r2_scores_catboost = []
rmse_scores_catboost = []
r2_scores_rf = []
rmse_scores_rf = []

# Dictionaries to hold feature importances
feature_importances_dict_lgb = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}
feature_importances_dict_cat = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}
perm_importances_dict_lgb = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}
perm_importances_dict_cat = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}
feature_importances_dict_rf = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}
perm_importances_dict_rf = {feature: [] for feature in x_columns + ['Oad1_Inc1', 'Oad2_Inc2']}

# Loop through each Excel file
for file_path in file_paths:
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Feature engineering: add interaction features
    df['Oad1_Inc1'] = df['Oad1'] * df['Inc1']
    df['Oad2_Inc2'] = df['Oad2'] * df['Inc2']

    # Update feature columns
    x_columns_extended = x_columns + ['Oad1_Inc1', 'Oad2_Inc2']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[x_columns_extended], df[y_column], random_state=42)

    # LightGBM model
    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    param_grid_lgb = {
        'num_leaves': [31, 40, 50],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300]
    }
    grid_search_lgb = GridSearchCV(estimator=lgb_model, param_grid=param_grid_lgb, cv=5, n_jobs=-1, scoring='r2')
    grid_search_lgb.fit(X_train, y_train)
    best_model_lgb = grid_search_lgb.best_estimator_

    # Fit the model and predict
    best_model_lgb.fit(X_train, y_train)
    y_pred_lgb = best_model_lgb.predict(X_test)

    # Calculate R² and RMSE for LightGBM
    r2_lgb = r2_score(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_scores_lightgbm.append(r2_lgb)
    rmse_scores_lightgbm.append(rmse_lgb)

    # Get feature importances for LightGBM
    feature_importances_lgb = best_model_lgb.feature_importances_
    for feature, importance in zip(x_columns_extended, feature_importances_lgb):
        feature_importances_dict_lgb[feature].append(importance)

    # Permutation importance for LightGBM
    perm_importance_lgb = permutation_importance(best_model_lgb, X_test, y_test, n_repeats=10, random_state=42)
    for feature, importance in zip(x_columns_extended, perm_importance_lgb.importances_mean):
        perm_importances_dict_lgb[feature].append(importance)

    # CatBoost model
    cat_model = CatBoostRegressor(random_seed=42, verbose=0)
    param_grid_cat = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'iterations': [100, 200, 300]
    }
    grid_search_cat = GridSearchCV(estimator=cat_model, param_grid=param_grid_cat, cv=5, n_jobs=-1, scoring='r2')
    grid_search_cat.fit(X_train, y_train)
    best_model_cat = grid_search_cat.best_estimator_

    # Fit the model and predict
    best_model_cat.fit(X_train, y_train)
    y_pred_cat = best_model_cat.predict(X_test)

    # Calculate R² and RMSE for CatBoost
    r2_cat = r2_score(y_test, y_pred_cat)
    rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
    r2_scores_catboost.append(r2_cat)
    rmse_scores_catboost.append(rmse_cat)

    # Get feature importances for CatBoost
    feature_importances_cat = best_model_cat.get_feature_importance()
    for feature, importance in zip(x_columns_extended, feature_importances_cat):
        feature_importances_dict_cat[feature].append(importance)

    # Permutation importance for CatBoost
    perm_importance_cat = permutation_importance(best_model_cat, X_test, y_test, n_repeats=10, random_state=42)
    for feature, importance in zip(x_columns_extended, perm_importance_cat.importances_mean):
        perm_importances_dict_cat[feature].append(importance)

    # RandomForest model
    rf_model = RandomForestRegressor(random_state=42)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='r2')
    grid_search_rf.fit(X_train, y_train)
    best_model_rf = grid_search_rf.best_estimator_

    # Fit the model and predict
    best_model_rf.fit(X_train, y_train)
    y_pred_rf = best_model_rf.predict(X_test)

    # Calculate R² and RMSE for RandomForest
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_scores_rf.append(r2_rf)
    rmse_scores_rf.append(rmse_rf)

    # Get feature importances for RandomForest
    feature_importances_rf = best_model_rf.feature_importances_
    for feature, importance in zip(x_columns_extended, feature_importances_rf):
        feature_importances_dict_rf[feature].append(importance)

    # Permutation importance for RandomForest
    perm_importance_rf = permutation_importance(best_model_rf, X_test, y_test, n_repeats=10, random_state=42)
    for feature, importance in zip(x_columns_extended, perm_importance_rf.importances_mean):
        perm_importances_dict_rf[feature].append(importance)

# Print summary of all results
print("LightGBM Results:")
summary_df_lgb = pd.DataFrame({
    'File': file_paths,
    'R²': r2_scores_lightgbm,
    'RMSE': rmse_scores_lightgbm
})
print(summary_df_lgb)

print("\nCatBoost Results:")
summary_df_cat = pd.DataFrame({
    'File': file_paths,
    'R²': r2_scores_catboost,
    'RMSE': rmse_scores_catboost
})
print(summary_df_cat)

print("\nRandomForest Results:")
summary_df_rf = pd.DataFrame({
    'File': file_paths,
    'R²': r2_scores_rf,
    'RMSE': rmse_scores_rf
})
print(summary_df_rf)

# Save the summary results to CSV
summary_df_lgb.to_csv('lightgbm_summary_results.csv', index=False)
summary_df_cat.to_csv('catboost_summary_results.csv', index=False)
summary_df_rf.to_csv('randomforest_summary_results.csv', index=False)

# Create dataframes for feature importances
feature_importances_df_lgb = pd.DataFrame(feature_importances_dict_lgb, index=file_paths)
perm_importances_df_lgb = pd.DataFrame(perm_importances_dict_lgb, index=file_paths)
feature_importances_df_cat = pd.DataFrame(feature_importances_dict_cat, index=file_paths)
perm_importances_df_cat = pd.DataFrame(perm_importances_dict_cat, index=file_paths)
feature_importances_df_rf = pd.DataFrame(feature_importances_dict_rf, index=file_paths)
perm_importances_df_rf = pd.DataFrame(perm_importances_dict_rf, index=file_paths)

# Plot heatmap for feature importances
plt.figure(figsize=(12, 8))
sns.heatmap(feature_importances_df_lgb.T, annot=True, cmap='coolwarm')
plt.title('LightGBM Feature Importance Heatmap')
plt.tight_layout()
plt.savefig('lightgbm_feature_importance_heatmap.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(perm_importances_df_lgb.T, annot=True, cmap='coolwarm')
plt.title('LightGBM Permutation Importance Heatmap')
plt.tight_layout()
plt.savefig('lightgbm_permutation_importance_heatmap.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(feature_importances_df_cat.T, annot=True, cmap='coolwarm')
plt.title('CatBoost Feature Importance Heatmap')
plt.tight_layout()
plt.savefig('catboost_feature_importance_heatmap.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(perm_importances_df_cat.T, annot=True, cmap='coolwarm')
plt.title('CatBoost Permutation Importance Heatmap')
plt.tight_layout()
plt.savefig('catboost_permutation_importance_heatmap.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(feature_importances_df_rf.T, annot=True, cmap='coolwarm')
plt.title('RandomForest Feature Importance Heatmap')
plt.tight_layout()
plt.savefig('randomforest_feature_importance_heatmap.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(perm_importances_df_rf.T, annot=True, cmap='coolwarm')
plt.title('RandomForest Permutation Importance Heatmap')
plt.tight_layout()
plt.savefig('randomforest_permutation_importance_heatmap.png')
plt.close()

# Save the feature importances to CSV
feature_importances_df_lgb.to_csv('lightgbm_feature_importances.csv')
perm_importances_df_lgb.to_csv('lightgbm_permutation_importances.csv')
feature_importances_df_cat.to_csv('catboost_feature_importances.csv')
perm_importances_df_cat.to_csv('catboost_permutation_importances.csv')
feature_importances_df_rf.to_csv('randomforest_feature_importances.csv')
perm_importances_df_rf.to_csv('randomforest_permutation_importances.csv')
