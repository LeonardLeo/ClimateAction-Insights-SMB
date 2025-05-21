# -*- coding: utf-8 -*-
"""
Created on Tue May 20 22:39:31 2025

@author: leona
"""

# =============================================================================
# Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import warnings
# from ydata_profiling import ProfileReport
from utils import (eda_x, 
                   safe_log1p_transform, 
                   breakdown_correlation_by_categories,
                   generate_correlation_summary,
                   run_causal_inference)



# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset 
# =============================================================================
# ---> Convert to Parquet File for Speed Processing
# data = pd.read_excel("data/deidentified_company_initiatives.xlsx")
# data.to_parquet("data/company_initiatives.parquet", index=False) # Save as parquet

# ---> Load Parquet Dataset
data = pd.read_parquet("data/company_initiatives.parquet")
dataset = pd.read_parquet("data/company_initiatives.parquet")
columns = dataset.columns

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
EDA = eda_x(dataset)
# EDA_data = eda_x(dataset, 
#                  graphs = True,
#                  barchart_figsize = (15, 10),
#                  max_categories = 15,
#                  pairplot = False,  # 👈 turn this off for large datasets
#                  drop_columns = ["company_id", "geography_id", "initiative_id"],
#                  save_graphs = True)

# =============================================================================
# EDA Visualization
# =============================================================================
# profile = ProfileReport(dataset, title="SMB Climate EDA", explorative=True)
# profile.to_file("eda_viz/climate_report.html")

# =============================================================================
# EDA - Correlation Analysis after Handling Skwed Distributions
# =============================================================================
# Finding the log of a column helps reduce the effect of outliers and can 
# be taken as first steps before correlatyion analysis.

columns_to_transform = [
    "turnover", 
    "employees", 
    "emissions_t_CO2e", 
    "initiative_carbon_savings_t_CO2e", 
    "initiative_financial_savings_GBP", 
    "initiative_capex_GBP"
]

df_transformed, transformation_log = safe_log1p_transform(data, 
                                                          columns_to_transform, 
                                                          drop_original_columns = True)

# Check transformation metadata
print(transformation_log)

correlation_1 = df_transformed.drop(["sector_code", "initiative_id"], axis = 1).corr(numeric_only=True, method = "spearman")

# =============================================================================
# EDA - Correlation Analysis for Various Groups
# =============================================================================
categorical_cols = [
    "geography_name",
    "industry_name",
    "initiative_categories",
    "sector_name",
    "initiative_implementation_difficulty",
    "initiative_operation_difficulty"
]

correlation_breakdown = breakdown_correlation_by_categories(
    df=df_transformed.drop(["sector_code", "initiative_id"], axis = 1),
    categorical_columns = categorical_cols,
    min_samples = 20,
    save_path = "correlation_summary/correlation_breakdown.pkl"
)

main_variables = df_transformed.drop(["sector_code", "initiative_id"], axis = 1).columns

summary_text = generate_correlation_summary(
    correlation_breakdown,
    main_vars = main_variables,
    corr_threshold = 0.5,
    include_all = False
)

# Print or save to a text file
print(summary_text)
with open("correlation_summary/correlation_summary_report.txt", "w") as f:
    f.write(summary_text)
    
# =============================================================================
# EDA - Causation Analysis
# =============================================================================
