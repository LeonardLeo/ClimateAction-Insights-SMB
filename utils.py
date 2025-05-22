# -*- coding: utf-8 -*-
"""
Created on Tue May 20 22:40:01 2025

@author: leona
"""
# =============================================================================
# Import Libraries
# =============================================================================
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, joblib  # For saving the dictionary as a pickle
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import f_regression, SelectFpr, SelectKBest
from sklearn.pipeline import make_pipeline

# =============================================================================
# Defining Functions
# =============================================================================
def breakdown_correlation_by_categories(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: Optional[List[str]] = None,
    min_samples: int = 10,
    save_path: Optional[str] = None,
    method_applied: str = "spearman"
) -> Dict:
    """
    For each categorical column, computes correlation matrices for each unique value (category)
    on the filtered data subset. Also records sample size used for each correlation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset.
    categorical_columns : List[str]
        List of categorical columns to break correlations down by.
    numeric_columns : Optional[List[str]]
        List of numeric columns to use for correlation. If None, auto-detects numeric cols.
    min_samples : int, default=10
        Minimum number of rows required to compute correlation for a category. Skips smaller.
    save_path : Optional[str]
        If provided, the nested dictionary is saved as a joblib pickle at this location.
    
    Returns:
    --------
    Dict
        Nested dictionary in format:
        {
          categorical_column: {
            category_value: {
                "sample_size": int,
                "correlation_matrix": pd.DataFrame
            },
            ...
          },
          ...
        }
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        
    result = {}
    
    for cat_col in categorical_columns:
        if cat_col not in df.columns:
            print(f"Warning: {cat_col} not found in dataframe columns. Skipping.")
            continue
        
        result[cat_col] = {}
        unique_values = df[cat_col].dropna().unique()
        
        for val in unique_values:
            subset = df[df[cat_col] == val]
            sample_size = len(subset)
            
            if sample_size < min_samples:
                # Skip small groups to avoid unreliable correlations
                continue
            
            corr_matrix = subset[numeric_columns].corr(numeric_only = True, method = method_applied)
            result[cat_col][val] = {
                "sample_size": sample_size,
                "correlation_matrix": corr_matrix
            }
    
    if save_path:
        joblib.dump(result, save_path)
        print(f"Saved correlation breakdown dictionary to {save_path}")
    
    return result

def generate_correlation_summary(
    correlation_dict: dict,
    main_vars: list,
    corr_threshold: float = 0.5,
    include_all: bool = False
) -> str:
    """
    Generates a summary report of correlations from the nested dictionary.

    Parameters:
    -----------
    correlation_dict : dict
        Output of breakdown_correlation_by_categories()
    main_vars : list
        List of variable names to highlight in correlation pairs.
    corr_threshold : float
        Threshold above which correlations are considered strong (absolute value).
    include_all : bool
        If True, include all correlations; if False, only strong ones.

    Returns:
    --------
    str
        Multi-line string report summarizing correlations.
    """
    lines = []
    for cat_col, categories in correlation_dict.items():
        lines.append(f"=== Breakdown by {cat_col} ===\n")
        for category, info in categories.items():
            sample_size = info["sample_size"]
            corr_matrix = info["correlation_matrix"]
            
            lines.append(f"Category: {category} (sample size: {sample_size})")
            
            # Find correlations between main_vars only (pairwise)
            reported_pairs = []
            for i, var1 in enumerate(main_vars):
                for var2 in main_vars[i+1:]:
                    if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
                        corr_val = corr_matrix.at[var1, var2]
                        if pd.isna(corr_val):
                            continue
                        if include_all or abs(corr_val) >= corr_threshold:
                            sign = "Positive" if corr_val > 0 else "Negative"
                            lines.append(f"  {var1} <-> {var2}: {corr_val:.3f} ({sign})")
                            reported_pairs.append((var1, var2, corr_val))
            
            if not reported_pairs:
                lines.append("  No correlations above threshold.")
            lines.append("")  # Empty line for spacing
    return "\n".join(lines)

def fix_string(word: str):
    import string
    punctuations = string.punctuation
    
    for each_letter in word:
        if each_letter in punctuations:
            word = word.replace(each_letter, " ")
    
    return word


def eda(dataset: pd.DataFrame,
        bin_size: int or list = None,
        graphs: bool = False,
        hue: str = None,
        markers: list = None,
        only_graphs: bool = False,
        hist_figsize: tuple = (15, 10),
        corr_heatmap_figsize: tuple = (15, 10),
        pairplot_figsize: tuple = (15, 10)) -> dict:

    if only_graphs != True:
        data_unique = {}
        data_category_count = {}
        data_numeric_count = {}
        dataset.info()
        data_head = dataset.head()
        data_tail = dataset.tail()
        data_mode = dataset.mode().iloc[0]
        data_descriptive_stats = dataset.describe()
        data_more_descriptive_stats = dataset.describe(include = "all")
        data_correlation_matrix = dataset.corr(numeric_only = True)
        data_distinct_count = dataset.nunique()
        data_count_duplicates = dataset.duplicated().sum()
        data_duplicates = dataset[dataset.duplicated()]
        data_count_null = dataset.isnull().sum()
        # data_null = dataset[any(dataset.isna())]
        data_total_null = dataset.isnull().sum().sum()
        for each_column in dataset.columns: # Loop through each column and get the unique values
            data_unique[each_column] = dataset[each_column].unique()
        for each_column in dataset.select_dtypes(object).columns:
            # Loop through the categorical columns and count how many values are in each category
            data_category_count[each_column] = dataset[each_column].value_counts()
        for each_column in dataset.select_dtypes(exclude = object).columns:
            # Loop through the numeric columns and count how many values are in each category
            data_numeric_count[each_column] = dataset[each_column].value_counts()

    if graphs == True:
        # Visualising Histograms
        dataset.hist(figsize = hist_figsize, bins = bin_size)
        plt.show()

        if only_graphs != False:
            # Creating a heatmap for the correlation matrix
            plt.figure(figsize = corr_heatmap_figsize)
            sns.heatmap(data_correlation_matrix, annot = True, cmap = 'coolwarm')
            plt.show()

        # Creating the pairplot for the dataset
        plt.figure(figsize = pairplot_figsize)
        sns.pairplot(dataset, hue = hue, markers = markers) # Graph of correlation across each numerical feature
        plt.show()

    if only_graphs != True:
        result = {"data_head": data_head,
                  "data_tail": data_tail,
                  "data_mode": data_mode,
                  "data_descriptive_stats": data_descriptive_stats,
                  "data_more_descriptive_stats": data_more_descriptive_stats,
                  "data_correlation_matrix": data_correlation_matrix,
                  "data_distinct_count": data_distinct_count,
                  "data_count_duplicates": data_count_duplicates,
                  "data_count_null": data_count_null,
                  "data_total_null": data_total_null,
                  "data_unique": data_unique,
                  "data_duplicates": data_duplicates,
                  # "data_null": data_null,
                  "data_category_count": data_category_count,
                  "data_numeric_count": data_numeric_count,
                  }
        return result

def eda_x(dataset: pd.DataFrame,
        bin_size: int or list = None,
        corr_method: str = "pearson",
        graphs: bool = False,
        pairplot: bool = False,  # New parameter
        max_categories: int = 20,  # You can parameterize this if you like
        save_graphs: bool = False,
        save_path: str = "eda_viz",
        hue: str = None,
        markers: list = None,
        only_graphs: bool = False,
        hist_figsize: tuple = (15, 10),
        corr_heatmap_figsize: tuple = (15, 10),
        barchart_figsize: tuple = (15, 10),
        pairplot_figsize: tuple = (15, 10),
        drop_columns: list = None,
        inplace_drop: bool = False) -> dict:

    
    # Handle dropped columns
    drop_columns = drop_columns or []
    filtered_dataset = dataset.drop(columns=drop_columns, errors='ignore')
    
    if save_graphs:
        os.makedirs(save_path, exist_ok=True)

    if only_graphs != True:
        data_unique = {}
        data_category_count = {}
        data_numeric_count = {}
        dataset.info()
        data_head = dataset.head()
        data_tail = dataset.tail()
        data_mode = dataset.mode().iloc[0]
        data_descriptive_stats = dataset.describe()
        data_more_descriptive_stats = dataset.describe(include="all")
        data_correlation_matrix = filtered_dataset.corr(numeric_only=True, method = corr_method)
        data_distinct_count = dataset.nunique()
        data_count_duplicates = dataset.duplicated().sum()
        data_duplicates = dataset[dataset.duplicated()]
        data_count_null = dataset.isnull().sum()
        data_total_null = dataset.isnull().sum().sum()

        for col in dataset.columns:
            data_unique[col] = dataset[col].unique()
        for col in dataset.select_dtypes(include="object").columns:
            data_category_count[col] = dataset[col].value_counts()
        for col in dataset.select_dtypes(exclude="object").columns:
            data_numeric_count[col] = dataset[col].value_counts()

    if graphs:
        # ===== HISTOGRAMS =====
        filtered_dataset.hist(figsize=hist_figsize, bins=bin_size)
        plt.suptitle("Histograms of Numeric Features", fontsize=16)
        if save_graphs:
            plt.savefig(f"{save_path}/histograms.png", bbox_inches="tight")
        plt.show()

        # ===== CORRELATION HEATMAP =====
        if only_graphs != False:
            plt.figure(figsize=corr_heatmap_figsize)
            sns.heatmap(data_correlation_matrix, annot=True, cmap='coolwarm')
            plt.title("Correlation Matrix", fontsize=16)
            if save_graphs:
                plt.savefig(f"{save_path}/correlation_matrix.png", bbox_inches="tight")
            plt.show()

        # ===== PAIRPLOT =====
        if pairplot:
            sns.pairplot(filtered_dataset, hue=hue, markers=markers)
            plt.suptitle("Pairplot of Numeric Features", y=1.02, fontsize=16)
            if save_graphs:
                plt.savefig(f"{save_path}/pairplot.png", bbox_inches="tight")
            plt.show()
            
        # ===== CATEGORICAL VALUE COUNTS ONE-BY-ONE WITH SKIP LIMIT =====
        categorical_cols = dataset.select_dtypes(include="object").columns
        plotted_columns = []
        skipped_columns = []
    
        for col in categorical_cols:
            num_unique = dataset[col].nunique()
            if num_unique > max_categories:
                skipped_columns.append(col)
                continue
    
            plt.figure(figsize=barchart_figsize)
            order = dataset[col].value_counts().index
            sns.countplot(data=dataset, x=col, order=order)
            plt.title(f"Count Plot of '{col}' ({num_unique} Categories)", fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
    
            if save_graphs:
                filename = os.path.join(save_path, f"countplot_{col}.png")
                plt.savefig(filename, bbox_inches="tight")
    
            plt.show()
            plotted_columns.append(col)
    
        # ===== PRINT SUMMARY =====
        print("\n✅ Categorical Columns Visualized:")
        print(plotted_columns)
        print("\n⚠️  Categorical Columns Skipped (too many unique values):")
        print(skipped_columns)


    if only_graphs != True:
        return {
            "data_head": data_head,
            "data_tail": data_tail,
            "data_mode": data_mode,
            "data_descriptive_stats": data_descriptive_stats,
            "data_more_descriptive_stats": data_more_descriptive_stats,
            "data_correlation_matrix": data_correlation_matrix,
            "data_distinct_count": data_distinct_count,
            "data_count_duplicates": data_count_duplicates,
            "data_count_null": data_count_null,
            "data_total_null": data_total_null,
            "data_unique": data_unique,
            "data_duplicates": data_duplicates,
            "data_category_count": data_category_count,
            "data_numeric_count": data_numeric_count
        }

def safe_log1p_transform(df: pd.DataFrame, 
                         columns: list, 
                         suffix: str = "_log1p",
                         drop_original_columns: bool = True) -> (pd.DataFrame, pd.DataFrame):
    """
    Apply log1p transformation to specified numeric columns,
    with automatic shifting if any value is <= -1.
    
    Parameters:
        df (pd.DataFrame): Original dataset
        columns (list): List of column names to transform
        suffix (str): Suffix to append to transformed columns
        
    Returns:
        df_transformed (pd.DataFrame): DataFrame with new log1p-transformed columns
        log_summary (pd.DataFrame): Summary of shifts applied to each column
    """
    
    df_transformed = df.copy()
    log_summary = []

    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Skipped: {col} not found in DataFrame.")
            continue

        if not np.issubdtype(df[col].dtype, np.number):
            print(f"⚠️ Skipped: {col} is not numeric.")
            continue

        col_min = df[col].min()

        if np.isnan(col_min):
            print(f"⚠️ Skipped: {col} contains only NaNs.")
            continue

        if col_min <= -1:
            shift = abs(col_min) + 1.1
            transformed = np.log1p(df[col] + shift)
            shift_applied = True
        else:
            shift = 0
            transformed = np.log1p(df[col])
            shift_applied = False

        new_col_name = f"{col}{suffix}"
        df_transformed[new_col_name] = transformed

        log_summary.append({
            "column": col,
            "new_column": new_col_name,
            "original_min": col_min,
            "shift_applied": shift_applied,
            "shift_value": shift if shift_applied else None
        })
    
    if drop_original_columns:
        df_transformed = df_transformed.drop(columns, axis = 1)

    log_df = pd.DataFrame(log_summary)
    return df_transformed, log_df

def data_preprocessing_pipeline(dataset: pd.DataFrame,
                                drop_columns: list = None,
                                log_col: list = None,
                                dummy_col: list = None,
                                replace_val: dict = None):
    # DATA CLEANING AND TRANSFORMATION
    # Converting the card type to numeric
    if replace_val is not None:
        dataset = dataset.replace(replace_val)
    
    # Fix dummy variables
    if dummy_col is not None:
        dataset = pd.get_dummies(dataset,
                                 columns = dummy_col,
                                 drop_first = True,
                                 dtype = np.int64)

    # Creating logrithmic columns
    if log_col is not None:
        for each_col in log_col:
            dataset[f"Log_{each_col}"] = np.log1p(dataset[each_col])

    # Dropping columns
    if drop_columns is not None:
        dataset = dataset.drop(drop_columns, axis = 1)

    return dataset

def run_regression_model(
    dataframe,
    X,
    y,
    model=None,
    model_name="Custom Model",
    test_size=0.2,
    standardize=False,
    log_y=False,
    log_feature_col=None,
    already_logged_y=False,
    cv_folds=5,
    random_state=42,
    plot=True,
    export_path=None,
    prediction_csv_path=None,
    return_metrics=False,
    remove_irrelevant_features=False,
    fpr_threshold=0.05,
    fallback_k=10,
    use_selectkbest=False,
    drop_columns_after_log=True,
    log_suffix="_log1p"
):
    log_feature_col = log_feature_col or []

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

    y_name = y.name if hasattr(y, "name") else "target"
    y = pd.Series(y, name=y_name)

    # Default model
    if model is None:
        model = LinearRegression()

    # Log transform target
    y_transformed = np.log1p(y) if log_y else y.copy()

    # Log transform selected features
    if len(log_feature_col) > 0:
        dataframe, _ = safe_log1p_transform(df=dataframe, columns=log_feature_col,
                                            suffix=log_suffix, drop_original_columns=drop_columns_after_log)
        X = dataframe.drop(y_name, axis = 1)  # Update X from transformed DataFrame
        feature_names = X.columns

    # Optional feature selection
    feature_scores_df = pd.DataFrame()
    if remove_irrelevant_features:
        print(f"\n🔎 Running SelectFpr with alpha={fpr_threshold}")
        fpr_selector = SelectFpr(score_func=f_regression, alpha=fpr_threshold)
        fpr_selector.fit(X, y_transformed)
        mask = fpr_selector.get_support()
        if mask.sum() == 0 or use_selectkbest:
            print(f"⚠️ No features passed FPR threshold. Falling back to top {fallback_k} with SelectKBest.")
            kbest_selector = SelectKBest(score_func=f_regression, k=min(fallback_k, X.shape[1]))
            kbest_selector.fit(X, y_transformed)
            mask = kbest_selector.get_support()
            scores = kbest_selector.scores_
            p_values = [np.nan] * len(scores)
            selector_name = "SelectKBest"
        else:
            scores = fpr_selector.scores_
            p_values = fpr_selector.pvalues_
            selector_name = "SelectFpr"

        selected_features = X.columns[mask]
        removed_features = X.columns[~mask]
        print(f"✅ {len(selected_features)} features selected using {selector_name}: {selected_features.tolist()}")
        if len(removed_features) > 0:
            print(f"🧹 Removed features: {removed_features.tolist()}")

        feature_scores_df = pd.DataFrame({
            "feature": feature_names,
            "score": scores,
            "p_value": p_values,
            "selected": mask
        }).sort_values(by="score", ascending=False)

        X = X.loc[:, mask]
        feature_names = X.columns

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=test_size, random_state=random_state
    )

    # Prepare pipeline if standardizing
    if standardize:
        pipeline = make_pipeline(StandardScaler(), model)
    else:
        pipeline = model

    # Fit pipeline
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Bias correction if log_y
    if log_y or already_logged_y:
        residuals_log = y_test - y_pred
        sigma2 = np.var(residuals_log)
        y_pred_corrected = np.expm1(y_pred + 0.5 * sigma2)
        y_test_eval = np.expm1(y_test)
        y_pred_eval = y_pred_corrected
        print(f"\n🧪 Bias Correction Applied: Mean Shift = {(y_pred_corrected - np.expm1(y_pred)).mean():.4f}")
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred

    # Evaluation metrics
    r2 = r2_score(y_test_eval, y_pred_eval)
    mse = mean_squared_error(y_test_eval, y_pred_eval)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_eval, y_pred_eval)

    print(f"\n📊 {model_name} Evaluation Metrics:")
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Cross-validation with same pipeline
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y_transformed, cv=kf, scoring='r2')
    print(f"\n🔁 {cv_folds}-Fold CV R² Scores: {cv_scores}")
    print(f"✅ CV Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Coefficients or importances
    model_used = pipeline.named_steps[model.__class__.__name__.lower()] if standardize else model
    coef_df = pd.DataFrame()
    if hasattr(model_used, "coef_"):
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": model_used.coef_
        }).sort_values(by="coefficient", key=abs, ascending=False)
        print("\n🧠 Feature Coefficients:")
        print(coef_df)
    elif hasattr(model_used, "feature_importances_"):
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model_used.feature_importances_
        }).sort_values(by="importance", ascending=False)
        print("\n🌲 Feature Importances:")
        print(coef_df)
    else:
        print("\nℹ️ This model does not expose feature coefficients/importances.")

    # Export model
    if export_path:
        if export_path.endswith(".joblib"):
            joblib.dump(pipeline, export_path)
            print(f"💾 Model saved to {export_path} (joblib)")
        elif export_path.endswith(".pkl"):
            with open(export_path, "wb") as f:
                pickle.dump(pipeline, f)
            print(f"💾 Model saved to {export_path} (pickle)")
        else:
            print("⚠️ export_path must end with .joblib or .pkl")

    # Save predictions
    if prediction_csv_path:
        pred_df = pd.DataFrame({
            "actual": y_test_eval,
            "predicted": y_pred_eval
        })
        pred_df.to_csv(prediction_csv_path, index=False)
        print(f"📁 Predictions saved to {prediction_csv_path}")

    # Plotting
    if plot:
        residuals = y_test_eval - y_pred_eval
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(residuals, kde=True, bins=30)
        plt.title("Residual Distribution")

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_test_eval, y=y_pred_eval)
        plt.plot([y_test_eval.min(), y_test_eval.max()],
                 [y_test_eval.min(), y_test_eval.max()], 'r--')
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.show()

    # Output
    metrics = {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "cv_mean_r2": cv_scores.mean(),
        "cv_std_r2": cv_scores.std()
    }
    data_output = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    return (pipeline, coef_df, metrics, feature_scores_df, data_output) if return_metrics else (pipeline, coef_df, data_output)



def one_hot_encode_with_rare_grouping(df, columns, threshold=0.01, drop_first=True):
    """
    Groups rare categories into 'Other' for the specified columns and applies one-hot encoding.

    Parameters:
    - df (pd.DataFrame): The input dataframe
    - columns (list): List of column names to encode
    - threshold (float): Minimum proportion to not be considered 'rare'
    - drop_first (bool): Whether to drop the first dummy column to avoid multicollinearity

    Returns:
    - pd.DataFrame: Transformed dataframe with one-hot encoded columns
    """
    df = df.copy()

    for col in columns:
        # Group rare categories into 'Other'
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        df[col] = df[col].apply(lambda x: 'Other' if x in rare_categories else x)

    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype = int)

    return df
