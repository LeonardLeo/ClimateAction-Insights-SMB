# =============================================================================
# Import Libraries
# =============================================================================
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import warnings
# from ydata_profiling import ProfileReport
from utils import (eda_x, 
                   one_hot_encode_with_rare_grouping, 
                   safe_log1p_transform,
                   breakdown_correlation_by_categories,
                   generate_correlation_summary)

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset 
# =============================================================================
# ---> Load Parquet Dataset
data = pd.read_parquet("../data/company_initiatives.parquet")
columns = data.columns

# =============================================================================
# Initial Exploratory Data Analysis
# =============================================================================
EDA = eda_x(data)
# EDA_data = eda_x(dataset, 
#                  graphs = True,
#                  barchart_figsize = (15, 10),
#                  max_categories = 15,
#                  pairplot = False,  # 👈 turn this off for large datasets
#                  drop_columns = ["company_id", "geography_id", "initiative_id"],
#                  save_graphs = True)

# ---> EDA Visualization
# profile = ProfileReport(dataset, title="SMB Climate EDA", explorative=True)
# profile.to_file("eda_viz/climate_report.html")

# =============================================================================
# Preprocessing Data
# =============================================================================
# ---> Dropping Irrelevant Features
company_data_processed = data.drop(["initiative_operation_difficulty",
                                    "initiative_implementation_difficulty",
                                    "initiative_id",
                                    "geography_id",
                                    "sector_code",
                                    "initiative_name",], axis = 1)

# Grouping and aggregating
group_cols = [
    "company_id", 
    "industry_name", 
    "sector_name",
    "geography_name",
    "turnover",
    "employees",
    "emissions_t_CO2e"
]

company_data_processed = (
    company_data_processed
    .groupby(group_cols)
    .agg(
        distinct_initiative_categories=("initiative_categories", pd.Series.nunique),
        total_initiative_carbon_savings_t_CO2e=("initiative_carbon_savings_t_CO2e", "sum"),
        total_initiative_financial_savings_GBP=("initiative_financial_savings_GBP", "sum"),
        total_initiative_capex_GBP=("initiative_capex_GBP", "sum"),
        total_initiatives=("initiative_categories", "count")  # 👈 count of all initiatives
    )
    .reset_index()
)

# Dropping Company_ID as it is also an irrelevant feature
processed_data = company_data_processed.drop("company_id", axis = 1)                                                    
# Dropping Sector_Name as industry will be used to provide context for where the SMB's belong
processed_data = processed_data.drop("sector_name", axis = 1) 

# =============================================================================
# EDA - Check Processed Data
# =============================================================================
EDA_processed_data = eda_x(processed_data, 
                           graphs = True,
                           barchart_figsize = (15, 10),
                           max_categories = 15,
                           pairplot = False,  # 👈 turn this off for large datasets
                           drop_columns = None,
                           save_graphs = True,
                           corr_method = "spearman",
                           save_path = "eda_processed_data")

# =============================================================================
# Data Cleaning Round-Up after EDA
# =============================================================================
# 122 duplicates found after data preprocessing.
processed_data = processed_data.drop_duplicates()

# ---> Handling Categorical Features
# Encoding Nominal Features
columns_to_encode = [
    'industry_name', 
    'geography_name', 
]

encoded_data = one_hot_encode_with_rare_grouping(
    processed_data,
    columns=columns_to_encode,
    threshold=0.01,       # Group categories < 1% as 'Other'
    drop_first=True       # Drop first to avoid dummy variable trap
)

# 5 duplicates found after encoding. Drop all duplicates
encoded_data = encoded_data.drop_duplicates()

# ---> Intoduce Logarithm to Skewed Columns
# Finding the log of a column helps reduce the effect of outliers and can 
# be taken as first steps before correlatyion analysis.
clean_data, _ = safe_log1p_transform(df = encoded_data,
                                     columns = ["turnover",
                                                "emissions_t_CO2e",
                                                "employees",
                                                "total_initiative_capex_GBP",
                                                "total_initiative_carbon_savings_t_CO2e",
                                                "total_initiative_financial_savings_GBP",
                                                "distinct_initiative_categories",
                                                "total_initiatives"])

# 101 duplicates found after data preprocessing. Drop all duplicates
clean_data = clean_data.drop_duplicates()

# =============================================================================
# EDA After Data Preprocessing
# =============================================================================
EDA_processed_data = eda_x(processed_data, 
                           graphs = True,
                           barchart_figsize = (15, 10),
                           max_categories = 15,
                           pairplot = False,  # 👈 turn this off for large datasets
                           drop_columns = None,
                           save_graphs = True,
                           corr_method = "spearman",
                           save_path = "eda_processed_data")

EDA_encoded_data = eda_x(encoded_data, 
                         graphs = True,
                         barchart_figsize = (15, 10),
                         hist_figsize = (50, 30),
                         max_categories = 15,
                         pairplot = False,  # 👈 turn this off for large datasets
                         drop_columns = None,
                         save_graphs = True,
                         corr_method = "spearman",
                         save_path = "eda_encoded_data")

EDA_clean_data = eda_x(clean_data, 
                       graphs = True,
                       barchart_figsize = (15, 10),
                       hist_figsize = (50, 30),
                       max_categories = 15,
                       pairplot = False,  # 👈 turn this off for large datasets
                       drop_columns = None,
                       save_graphs = True,
                       corr_method = "spearman",
                       save_path = "eda_clean_data")

# =============================================================================
# EDA - Correlation Analysis for Various Groups
# =============================================================================
categorical_cols = [
    "geography_name",
    "industry_name"
]

correlation_breakdown = breakdown_correlation_by_categories(
    df = processed_data,
    categorical_columns = categorical_cols,
    min_samples = 20,
    save_path = "correlation_summary/correlation_breakdown_preprocessed_data.pkl"
)

main_variables = processed_data.columns

summary_text = generate_correlation_summary(
    correlation_breakdown,
    main_vars = main_variables,
    corr_threshold = 0.5,
    include_all = False
)

# Print or save to a text file
print(summary_text)
with open("correlation_summary/correlation_preprocessed_data_summary_report.txt", "w") as f:
    f.write(summary_text)

# =============================================================================
# Save Preprocessed Dataset
# =============================================================================
processed_data.to_csv("../data/preprocessed_data/clean_data_not_encoded_and_not_log_transformed.csv", index = False)
processed_data.to_parquet("../data/preprocessed_data/clean_data_not_encoded_and_not_log_transformed.parquet", index = False)

clean_data.to_csv("../data/preprocessed_data/clean_data_encoded_and_log_transformed.csv", index = False)
clean_data.to_parquet("../data/preprocessed_data/clean_data_encoded_and_log_transformed.parquet", index = False)

encoded_data.to_csv("../data/preprocessed_data/clean_data_encoded_without_log_transform.csv", index = False)
encoded_data.to_parquet("../data/preprocessed_data/clean_data_encoded_without_log_transform.parquet", index = False)
