# =============================================================================
# Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import warnings
from company_info import get_summary_data

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset 
# =============================================================================
initiative_df = pd.read_parquet("../data_created/company_summary.parquet")

# =============================================================================
# Recommend Initiatives
# =============================================================================
def recommend_initiatives(initiative_df, profile, top_n = 5):
    summary_data = get_summary_data(profile['industry_name'], profile['geography_name'])
    
    filtered = initiative_df[
        (initiative_df['industry_name'] == profile['industry_name']) &
        (initiative_df['geography_name'] == profile['geography_name'])
    ].copy()
    
    if filtered.empty:
        return None, None

    # Median comparisons
    median_employees = filtered['employees'].median()
    median_emissions = filtered['emissions_t_CO2e'].median()
    
    size_factor = profile['employees'] / median_employees if median_employees > 0 else 1
    emission_factor = profile['emissions_t_CO2e'] / median_emissions if median_emissions > 0 else 1

    # Adjusted score
    filtered['adjusted_score'] = filtered['initiative_score'] * \
                                  (1 / (1 + abs(size_factor - 1))) * \
                                  (1 / (1 + abs(emission_factor - 1)))
    
    filtered['rank'] = filtered['adjusted_score'].rank(method='dense', ascending=False)

    # ✅ Select initiatives where rank is between 1 and top_n
    top_recommendations = filtered[(filtered['rank'] >= 1) & (filtered['rank'] <= top_n)]

    def get_possible_initiatives(recommendations):
        # Group data by initiatives
        grouped_data = top_recommendations.drop(
            ["sector_code", "industry_name", "geography_id", "geography_name", "initiative_id"], axis = 1
            ).groupby(
                "initiative_name"
            ).agg(
                avg_savings=pd.NamedAgg(column='initiative_financial_savings_GBP', aggfunc='mean'),
                avg_payback=pd.NamedAgg(column='payback_period', aggfunc='mean'),
                avg_capital_exp=pd.NamedAgg(column='initiative_capex_GBP', aggfunc='mean'),
                avg_financial_roi=pd.NamedAgg(column='financial_roi', aggfunc='mean'),
                avg_difficulty_score=pd.NamedAgg(column='difficulty_score', aggfunc='mean'),
                avg_carbon_savings=pd.NamedAgg(column='initiative_carbon_savings_t_CO2e', aggfunc='mean'),
                num_companies=pd.NamedAgg(column='company_id', aggfunc=lambda x: x.nunique())
            ).reset_index()
        return grouped_data
    
    initiatives = get_possible_initiatives(top_recommendations)
    return summary_data, filtered, top_recommendations, initiatives

# Example use case for function
smb_profile = {
    'industry_name': 'C: MANUFACTURING',
    'sector_name': 'Manufacture of electrical equipment',
    'geography_name': 'Westminster',
    'employees': 50,
    'turnover': 3_000_000,  # GBP
    'emissions_t_CO2e': 100
}
summary_data, filtered_df, top_recommendations, possible_initiatives = recommend_initiatives(initiative_df, smb_profile)
