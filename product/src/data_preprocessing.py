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
# Get Dataset 
# =============================================================================
initiative_df = pd.read_parquet("../../data/company_initiatives.parquet")
columns = initiative_df.columns
data_head = initiative_df.head()
data_implementation_difficulty = pd.Series(initiative_df["initiative_implementation_difficulty"].unique(), name = "Name")
data_operation_difficulty = pd.Series(initiative_df["initiative_operation_difficulty"].unique(), name = "Name")
data_industry = pd.Series(initiative_df["industry_name"].unique(), name = "Name")

# ============================================================================= 
# Feature Engineering – Robust Financial & Carbon Metrics
# =============================================================================
# === Flags for edge cases ===
initiative_df['is_zero_capex'] = initiative_df['initiative_capex_GBP'] == 0
initiative_df['is_zero_savings'] = initiative_df['initiative_financial_savings_GBP'] == 0

# === Calculate financial ROI ===
initiative_df['financial_roi'] = np.where(
    initiative_df['is_zero_capex'],
    np.inf,  # Infinite ROI for free initiatives
    initiative_df['initiative_financial_savings_GBP'] / initiative_df['initiative_capex_GBP']
)

# === Calculate payback period ===
initiative_df['payback_period'] = np.where(
    initiative_df['is_zero_savings'],
    np.inf,  # Infinite payback period if no savings
    initiative_df['initiative_capex_GBP'] / initiative_df['initiative_financial_savings_GBP']
)

# === Carbon savings per £ invested ===
initiative_df['carbon_savings_per_£'] = np.where(
    initiative_df['is_zero_capex'],
    np.nan,  # undefined if no investment
    initiative_df['initiative_carbon_savings_t_CO2e'] / initiative_df['initiative_capex_GBP']
)

# === Clip extreme values to avoid skew ===
initiative_df['financial_roi'] = initiative_df['financial_roi'].replace(np.inf, 100).clip(lower=0, upper=100)
initiative_df['payback_period'] = initiative_df['payback_period'].replace(np.inf, 100).clip(lower=0, upper=100)
initiative_df['carbon_savings_per_£'] = initiative_df['carbon_savings_per_£'].clip(lower=0, upper=100)

# === Difficulty levels mapping (higher = easier) ===
difficulty_map = {
    'Negligible': 1.0,
    'Minimal': 0.85,
    'Minor': 0.7,
    'Moderate': 0.55,
    'Significant': 0.4,
    'Substantial': 0.25,
    'Extreme': 0.1
}

# Map and combine difficulties
initiative_df['implementation_difficulty_score'] = initiative_df['initiative_implementation_difficulty'].map(difficulty_map)
initiative_df['operation_difficulty_score'] = initiative_df['initiative_operation_difficulty'].map(difficulty_map)
initiative_df['difficulty_score'] = (
    initiative_df['implementation_difficulty_score'] + 
    initiative_df['operation_difficulty_score']
) / 2

# === Updated scoring function ===
def score_initiative(row):
    roi_score = row['financial_roi'] / 100
    payback_score = 1 - (row['payback_period'] / 100)
    carbon_score = row['carbon_savings_per_£'] / 100 if not np.isnan(row['carbon_savings_per_£']) else 0
    difficulty_score = row['difficulty_score'] if not pd.isna(row['difficulty_score']) else 0.5
    
    total_score = 0.4 * roi_score + 0.2 * payback_score + 0.2 * carbon_score + 0.2 * difficulty_score
    return total_score

initiative_df['initiative_score'] = initiative_df.apply(score_initiative, axis=1)
# === Dense ranking for ties ===
initiative_df['rank_within_company'] = initiative_df.groupby('company_id')['initiative_score']\
    .rank(method='dense', ascending=False)
    

# SAVE DATAFRAME
initiative_df.to_csv("../data_created/company_summary.csv")
initiative_df.to_parquet("../data_created/company_summary.parquet")