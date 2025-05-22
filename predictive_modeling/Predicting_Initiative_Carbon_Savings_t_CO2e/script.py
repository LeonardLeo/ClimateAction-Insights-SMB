# =============================================================================
# Import Libraries
# =============================================================================
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
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

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset 
# =============================================================================
# ---> Load Parquet Dataset
data = pd.read_parquet("../../data/preprocessed_data/clean_data_encoded_and_log_transformed.parquet")
columns = data.columns

# =============================================================================
# Build Model
# =============================================================================
regression_models = {
    "Linear Regression": LinearRegression(),
    # "Ridge Regression": Ridge(alpha=1.0),
    # "Lasso Regression": Lasso(alpha=0.1),
    # "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    # "Huber Regressor": HuberRegressor(epsilon=1.35),
    # "Bayesian Ridge": BayesianRidge(),
    # "Decision Tree": DecisionTreeRegressor(max_depth=5),
    # "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    # "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    # "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, random_state=42),
    # "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    # "SVR (RBF Kernel)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    # "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = {}
for name, model in regression_models.items():
    print(f"\n🧪 Running: {name}")
    each_model = run_regression_model(dataframe = data,
                                      X = data.drop("initiative_carbon_savings_t_CO2e_log1p", axis = 1),
                                      y = data.initiative_carbon_savings_t_CO2e_log1p,
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
    