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
initiative_df = pd.read_parquet("../data/company_initiatives.parquet")
columns = initiative_df.columns
data_head = initiative_df.head()
data_implementation_difficulty = pd.Series(initiative_df["initiative_implementation_difficulty"].unique(), name = "Name")
data_operation_difficulty = pd.Series(initiative_df["initiative_operation_difficulty"].unique(), name = "Name")
data_industry = pd.Series(initiative_df["industry_name"].unique(), name = "Name")

# ============================================================================= 
# Feature Engineering – Robust Financial & Carbon Metrics
# =============================================================================
import numpy as np

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

# -------------------------------------------
def get_top_recommendations(df, company_id, top_n=5):
    # Filter initiatives for the company
    company_inits = df[df['company_id'] == company_id].copy()
    
    # Sort by rank within company
    company_inits = company_inits.sort_values(by='rank_within_company')
    
    # Select top N
    top_inits = company_inits.head(top_n)
    
    # Format difficulty text from score (optional)
    def difficulty_text(score):
        if score > 0.85:
            return 'Negligible'
        elif score > 0.7:
            return 'Minimal'
        elif score > 0.55:
            return 'Minor'
        elif score > 0.4:
            return 'Moderate'
        elif score > 0.25:
            return 'Significant'
        elif score > 0.1:
            return 'Substantial'
        else:
            return 'Extreme'
    
    top_inits['difficulty_level'] = top_inits['difficulty_score'].apply(difficulty_text)
    
    # Create a user-friendly display
    recommendations = []
    for _, row in top_inits.iterrows():
        rec = {
            'Initiative': row['initiative_name'],
            'Category': row['initiative_categories'],
            'Estimated Carbon Savings (tCO2e)': round(row['initiative_carbon_savings_t_CO2e'], 2),
            'Financial ROI': f"{row['financial_roi']:.2f}x",
            'Payback Period (years)': f"{row['payback_period']:.2f}",
            'Difficulty': row['difficulty_level'],
            'Rank': int(row['rank_within_company']),
            'Score': round(row['initiative_score'], 3)
        }
        recommendations.append(rec)
        
    return recommendations

company_id_example = 'c4a9e0f4f38563f549c762683b04e07c'
top_recs = get_top_recommendations(initiative_df, company_id_example, top_n=3)

import pprint
pprint.pprint(top_recs)

# -----------------------------------------------
# Example SMB profile info collected from user input form
smb_profile = {
    'industry_name': 'C: MANUFACTURING',
    'sector_name': 'Manufacture of electrical equipment',
    'geography_name': 'Westminster',
    'employees': 50,
    'turnover': 3_000_000,  # GBP
    'emissions_t_CO2e': 100
}
	
# Aggregate summary stats per industry & geography
industry_summary = initiative_df.groupby('industry_name').agg({
    'initiative_carbon_savings_t_CO2e': 'mean',
    'initiative_financial_savings_GBP': 'mean',
    'initiative_capex_GBP': 'mean',
    'financial_roi': 'mean',
    'payback_period': 'mean',
    'initiative_score': 'mean'
}).reset_index()

geo_summary = initiative_df.groupby('geography_name').agg({
    'initiative_carbon_savings_t_CO2e': 'mean',
    'initiative_financial_savings_GBP': 'mean',
    'initiative_capex_GBP': 'mean',
    'financial_roi': 'mean',
    'payback_period': 'mean',
    'initiative_score': 'mean'
}).reset_index()

# Fetch industry stats for SMB's industry
ind_stats = industry_summary[industry_summary['industry_name'] == smb_profile['industry_name']].iloc[0]
geo_stats = geo_summary[geo_summary['geography_name'] == smb_profile['geography_name']].iloc[0]

print(f"In the {smb_profile['industry_name']} industry, SMBs reduce on average {ind_stats['initiative_carbon_savings_t_CO2e']:.2f} tCO2e annually through climate initiatives.")
print(f"In the {smb_profile['geography_name']} region, SMBs achieve average financial ROI of {geo_stats['financial_roi']:.2f} from such initiatives.")

def recommend_initiatives(df, profile, top_n=10):
    filtered = df[
        (df['industry_name'] == profile['industry_name']) &
        (df['geography_name'] == profile['geography_name'])
    ].copy()
    
    # Calculate size & emission factors vs median in filtered data
    median_employees = filtered['employees'].median() if 'employees' in filtered else profile['employees']
    median_emissions = filtered['emissions_t_CO2e'].median() if 'emissions_t_CO2e' in filtered else profile['emissions_t_CO2e']
    
    size_factor = profile['employees'] / median_employees if median_employees > 0 else 1
    emission_factor = profile['emissions_t_CO2e'] / median_emissions if median_emissions > 0 else 1
    
    # Adjust initiative score by closeness of SMB profile to initiative typical profile
    filtered['adjusted_score'] = filtered['initiative_score'] * (1 / (1 + abs(size_factor - 1))) * (1 / (1 + abs(emission_factor - 1)))
    
    filtered['rank'] = filtered['adjusted_score'].rank(method='dense', ascending=False)
    
    return filtered.sort_values('rank').head(top_n)

top_recommendations = recommend_initiatives(initiative_df, smb_profile)
print(top_recommendations[['initiative_name', 'initiative_score', 'adjusted_score', 'rank']])

def display_recommendations(recommendations, profile):
    print(f"\nTop initiative recommendations for your SMB in {profile['industry_name']} / {profile['geography_name']}:\n")
    for i, row in recommendations.iterrows():
        print(f"- {row['initiative_name']}:")
        print(f"  Estimated CO2 savings: {row['initiative_carbon_savings_t_CO2e']:.2f} tCO2e/year")
        print(f"  Estimated annual financial savings: £{row['initiative_financial_savings_GBP']:.2f}")
        print(f"  Estimated upfront cost: £{row['initiative_capex_GBP']:.2f}")
        print(f"  Financial ROI Score: {row['financial_roi']:.2f}")
        print(f"  Implementation Difficulty: {row['initiative_implementation_difficulty']}")
        print(f"  Operation Difficulty: {row['initiative_operation_difficulty']}")
        print(f"  Overall Initiative Score: {row['initiative_score']:.2f}\n")

display_recommendations(top_recommendations, smb_profile)

quick_wins = top_recommendations[
    (top_recommendations['initiative_capex_GBP'] < 5000) & 
    (top_recommendations['initiative_score'] > 0.6)
]

print("Quick Wins with low upfront investment:")
print(quick_wins[['initiative_name', 'initiative_capex_GBP', 'initiative_score']])

# Mapping for industry codes to clean names
industry_map = {
    "M: PROFESSIONAL AND SCIENTIFIC AND TECHNICAL ACTIVITIES": "Professional, Scientific & Technical Activities",
    "N: ADMINISTRATIVE AND SUPPORT SERVICE ACTIVITIES": "Administrative and Support Services",
    "G: WHOLESALE AND RETAIL TRADE; REPAIR OF MOTOR VEHICLES AND MOTORCYCLES": "Wholesale and Retail Trade",
    "C: MANUFACTURING": "Manufacturing",
    "Q: HUMAN HEALTH AND SOCIAL WORK ACTIVITIES": "Human Health and Social Work",
    "J: INFORMATION AND COMMUNICATION": "Information and Communication",
    "H: TRANSPORTATION AND STORAGE": "Transportation and Storage",
    "I: ACCOMMODATION AND FOOD SERVICE ACTIVITIES": "Accommodation and Food Services",
    "E: WATER SUPPLY; SEWERAGE AND WASTE MANAGEMENT AND REMEDIATION ACTIVITIES": "Water Supply & Waste Management",
    "F: CONSTRUCTION": "Construction",
    "R: ARTS AND ENTERTAINMENT AND RECREATION": "Arts, Entertainment & Recreation",
    "L: REAL ESTATE ACTIVITIES": "Real Estate Activities",
    "B: MINING AND QUARRYING": "Mining and Quarrying",
    "P: EDUCATION": "Education",
    "D: ELECTRICITY GAS STEAM AND AIR CONDITIONING SUPPLY": "Electricity & Gas Supply",
    "S: OTHER SERVICE ACTIVITIES": "Other Services",
    "A: AGRICULTURE FORESTRY AND FISHING": "Agriculture, Forestry & Fishing",
    "K: FINANCIAL AND INSURANCE ACTIVITIES": "Financial and Insurance Activities",
    "U: ACTIVITIES OF EXTRATERRITORIAL ORGANISATIONS AND BODIES": "Extraterritorial Organizations",
    "T: ACTIVITIES OF HOUSEHOLDS AS EMPLOYERS; UNDIFFERENTIATED GOODS- AND SERVICES-PRODUCING ACTIVITIES OF HOUSEHOLDS FOR OWN USE": "Household Activities"
}

def format_payback(payback):
    if payback == 0 or payback < 0.01:
        return "immediate"
    else:
        return f"{payback:.1f} years"

def filter_initiatives_for_company(initiatives_df, company_industry, company_location):
    return initiatives_df[
        (initiatives_df['industry_name'] == company_industry) &
        (initiatives_df['geography_name'] == company_location)
    ]

def summarize_initiatives(initiatives_df):
    # Group by name + context (industry and geography)
    grouped = initiatives_df.groupby(
        ['initiative_name', 'industry_name', 'geography_name']
    ).agg(
        avg_savings=pd.NamedAgg(column='initiative_financial_savings_GBP', aggfunc='mean'),
        avg_payback=pd.NamedAgg(column='payback_period', aggfunc='mean'),
        num_companies=pd.NamedAgg(column='company_id', aggfunc=lambda x: x.nunique())
    ).reset_index()

    # Sort by average savings (or another preferred metric)
    grouped = grouped.sort_values(by='avg_savings', ascending=False)

    return grouped

def get_top_initiatives_summary(initiatives_df, top_n=3):
    # Summarize initiatives stats
    summary_df = summarize_initiatives(initiatives_df)
    
    # Take top N initiatives by average savings
    top_initiatives = summary_df.head(top_n)
    
    return top_initiatives

def build_initiative_list_text(grouped_initiatives):
    lines = []
    for _, row in grouped_initiatives.iterrows():
        payback_str = format_payback(row['avg_payback'])
        savings_rounded = round(row['avg_savings'], 1)
        line = (f"- {row['initiative_name']}: Estimated average savings of £{savings_rounded} annually, "
                f"with an average payback period of {payback_str}. "
                f"This initiative has been adopted by {row['num_companies']} similar businesses in your area and industry.")
        lines.append(line)
    return "\n".join(lines)

def generate_email_message(company_name, company_industry, company_location,
                           avg_co2_reduction, avg_financial_roi,
                           top_initiatives_df):
    # Map raw industry codes to clean names (example)
    industry_map = {
        "C: MANUFACTURING": "Manufacturing",
        # Add all relevant mappings here...
    }
    clean_industry_name = industry_map.get(company_industry, company_industry)

    # Start email body
    email = f"Hi {company_name},\n\n"
    email += f"As a business in the {clean_industry_name} industry located in {company_location}, " \
             "you have a tremendous opportunity to reduce your carbon footprint while saving costs.\n\n"

    email += f"On average, companies like yours in the {clean_industry_name} sector reduce around " \
             f"{avg_co2_reduction:.1f} tonnes of CO2 annually through simple initiatives. " \
             f"In {company_location}, businesses achieve an average financial ROI of {avg_financial_roi:.1f} by adopting these climate actions.\n\n"

    email += "Based on your company profile, here are the top climate initiatives you could consider implementing:\n\n"

    for _, row in top_initiatives_df.iterrows():
        payback_desc = "immediate" if row['avg_payback'] <= 0 else f"{row['avg_payback']:.1f} years"
        email += f"- {row['initiative_name']}: Estimated average savings of £{row['avg_savings']:.1f} annually, " \
                 f"with an average payback period of {payback_desc}. " \
                 f"This initiative has been adopted by {row['num_companies']} similar businesses in your area and industry.\n"

    email += "\nTaking action not only benefits the environment but also your bottom line. Let’s get started on making a positive impact together!\n\n"
    email += "Best regards,\nThe Climate Action Team"

    return email

# Example function to get and email recommendations

def create_recommendation_email(initiatives_df, company_info, top_n=3):
    filtered = filter_initiatives_for_company(initiatives_df, company_info['industry_name'], company_info['geography_name'])
    top_inits = get_top_initiatives_summary(filtered, top_n)

    # Example: calculate averages for CO2 and ROI across filtered data
    avg_co2 = filtered['initiative_carbon_savings_t_CO2e'].mean() if not filtered.empty else 0
    avg_roi = filtered['financial_roi'].mean() if not filtered.empty else 0

    email_msg = generate_email_message(
        company_name=company_info['name'],
        company_industry=company_info['industry_name'],
        company_location=company_info['geography_name'],
        avg_co2_reduction=avg_co2,
        avg_financial_roi=avg_roi,
        top_initiatives_df=top_inits
    )
    return email_msg

# Usage:
company_info = {
    'name': 'ACME Ltd',
    'industry_name': 'C: MANUFACTURING',
    'geography_name': 'Westminster',
    # other info fields...
}

email_content = create_recommendation_email(initiative_df, company_info)
print(email_content)
