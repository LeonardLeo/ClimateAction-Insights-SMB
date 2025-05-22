# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:36:45 2025

@author: leona
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, BayesianRidge
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from utils import run_regression_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import f_regression, SelectFpr, SelectKBest
from sklearn.pipeline import make_pipeline

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset 
# =============================================================================
# ---> Load Parquet Dataset
data = pd.read_parquet("../../data/preprocessed_data/clean_data.parquet")
dataset = pd.read_parquet("../../data/preprocessed_data/clean_data.parquet")

data_1 = pd.read_parquet("../../data/preprocessed_data/clean_data_without_log_transform.parquet")
dataset_1 = pd.read_parquet("../../data/preprocessed_data/clean_data_without_log_transform.parquet")

columns = dataset.columns

# # =============================================================================
# # Split Dataset
# # =============================================================================
# X = dataset.drop("initiative_financial_savings_GBP_log1p", axis = 1)
# y = dataset.initiative_financial_savings_GBP_log1p

# # =============================================================================
# # Select Features - ANOVA STATISTICAL APPROACH
# # =============================================================================#
# feature_scores_df = pd.DataFrame()
# feature_names = X.columns

# # ---> Using SelectFPR
# fpr_selector = SelectFpr(score_func=f_regression)
# fpr_selector.fit(X, y)
# mask = fpr_selector.get_support()

# # ---> Get Scores and P-Value
# scores = fpr_selector.scores_
# p_values = fpr_selector.pvalues_
# selector_name = "SelectFpr"

# # ---> Information
# selected_features = X.columns[mask]
# removed_features = X.columns[~mask]
# print(f"✅ {len(selected_features)} features selected using {selector_name}: {selected_features.tolist()}")
# if len(removed_features) > 0:
#     print(f"🧹 Removed features: {removed_features.tolist()}")

# # ---> Output
# feature_scores_df = pd.DataFrame({
#     "feature": feature_names,
#     "score": scores,
#     "p_value": p_values,
#     "selected": mask
# }).sort_values(by="score", ascending=False)

# # Feature Selected X
# selected_X = X[selected_features.to_list()]
# print(selected_X)

# # =============================================================================
# # Train Test Split
# # =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(
#     selected_X, y, test_size=0.2, random_state=42
# )

# # =============================================================================
# # Model Training
# # =============================================================================
# # ---> Build Training Model
# regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# model = regressor.fit(X_train, y_train)

# # ---> Make Predictions and Get Error in Prediction
# # Model Prediction (Training)
# y_pred = model.predict(X_train)

# # Model Prediction (Test)
# y_pred_1 = model.predict(X_test)

# # ---> Model Evaluation
# # Training
# r_squared = r2_score(y_train, y_pred)

# # Test
# r_squared_1 = r2_score(y_test, y_pred_1)












# =============================================================================
# Build Model
# =============================================================================
regression_models = {
    # "Linear Regression": LinearRegression(),
    # "Ridge Regression": Ridge(alpha=1.0),
    # "Lasso Regression": Lasso(alpha=0.1),
    # "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    # "Huber Regressor": HuberRegressor(epsilon=1.35),
    # "Bayesian Ridge": BayesianRidge(),
    # "Decision Tree": DecisionTreeRegressor(max_depth=5),
    # "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    # "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, random_state=42),
    # "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    # "SVR (RBF Kernel)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    # "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = {}
for name, model in regression_models.items():
    print(f"\n🧪 Running: {name}")
    each_model = run_regression_model(dataframe = dataset_1,
                                      X = dataset.drop("initiative_financial_savings_GBP_log1p", axis = 1),
                                      y = dataset.initiative_financial_savings_GBP_log1p,
                                      # already_logged_y = True,
                                      # log_y = False,
                                      cv_folds = 10,
                                      export_path = f"results/model/{name}_model.pkl",
                                      prediction_csv_path = f"results/model_result/{name}_results.csv",
                                      return_metrics = True,
                                      model = model, 
                                      model_name = name,
                                      test_size = 0.2,
                                      standardize = True,
                                      plot = True,
                                      remove_irrelevant_features = True,
                                      fpr_threshold = 0.05,
                                      fallback_k = 10,
                                      use_selectkbest = False,
                                      drop_columns_after_log = True,
                                      log_suffix = "_log1p",
                                      log_feature_col = None)
    results[f"{name}"] = each_model
    
    
    