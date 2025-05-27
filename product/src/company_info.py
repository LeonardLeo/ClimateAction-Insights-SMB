# =============================================================================
# Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import warnings

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Function
# =============================================================================
# -----------------------------------------------
def get_summary_data(industry_name, geography_name):
    # Get Dataset 
    initiative_df = pd.read_parquet("../data_created/company_summary.parquet")
    
    # Aggregate summary stats per industry & geography
    industry_summary = initiative_df.groupby('industry_name').agg({
        "employees": "mean",
        "turnover": "mean",
        "emissions_t_CO2e": "mean",
        'initiative_carbon_savings_t_CO2e': 'mean',
        'initiative_financial_savings_GBP': 'mean',
        'initiative_capex_GBP': 'mean',
        'financial_roi': 'mean',
        'payback_period': 'mean',
        'initiative_score': 'mean'
    }).reset_index()
    
    geo_summary = initiative_df.groupby('geography_name').agg({
        "employees": "mean",
        "turnover": "mean",
        "emissions_t_CO2e": "mean",
        'initiative_carbon_savings_t_CO2e': 'mean',
        'initiative_financial_savings_GBP': 'mean',
        'initiative_capex_GBP': 'mean',
        'financial_roi': 'mean',
        'payback_period': 'mean',
        'initiative_score': 'mean'
    }).reset_index()
    
    ind_geo_summary = initiative_df.groupby(['industry_name', 'geography_name']).agg({
        "employees": "mean",
        "turnover": "mean",
        "emissions_t_CO2e": "mean",
        'initiative_carbon_savings_t_CO2e': 'mean',
        'initiative_financial_savings_GBP': 'mean',
        'initiative_capex_GBP': 'mean',
        'financial_roi': 'mean',
        'payback_period': 'mean',
        'initiative_score': 'mean'
    }).reset_index()
    
    # Fetch stats
    ind_stats = industry_summary[industry_summary['industry_name'] == industry_name].iloc[0]
    geo_stats = geo_summary[geo_summary['geography_name'] == geography_name].iloc[0]
    ind_and_geo_stats = ind_geo_summary[(ind_geo_summary['industry_name'] == industry_name) & (ind_geo_summary['geography_name'] == geography_name)]
        
    return initiative_df, ind_stats, geo_stats, ind_and_geo_stats


# # Example SMB profile info collected from user input form
# smb_profile = {
#     'industry_name': 'C: MANUFACTURING',
#     'sector_name': 'Manufacture of electrical equipment',
#     'geography_name': 'Westminster'
# }

# summary = get_summary_data(smb_profile['industry_name'], smb_profile['geography_name'])