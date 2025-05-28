# =============================================================================
# Import Libraries
# =============================================================================
import pandas as pd
import warnings
import argparse
from recommend_initiatives import recommend_initiatives

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Email Messenger
# =============================================================================
def generate_cold_email(profile):
    # Industry code to clean name mapping
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

    # Load data
    initiative_df = pd.read_parquet("../data_created/company_summary.parquet")
    summary_data, filtered_df, top_recommendations, possible_initiatives = recommend_initiatives(initiative_df, profile)
    ind_stats = summary_data[1]
    geo_stats = summary_data[2]
    ind_geo_stats = summary_data[3]

    # Profile info
    industry = profile['industry_name']
    geography = profile['geography_name']
    company_name = profile["company_name"]

    # Extract localised stats
    carbon_savings = round(ind_geo_stats['initiative_carbon_savings_t_CO2e'].values[0], 1)
    financial_savings = round(ind_geo_stats['initiative_financial_savings_GBP'].values[0], 2)
    capex = round(ind_geo_stats['initiative_capex_GBP'].values[0], 2)
    roi = round(ind_geo_stats['financial_roi'].values[0], 2)
    payback = round(ind_geo_stats['payback_period'].values[0], 2)
    score = round(ind_geo_stats['initiative_score'].values[0], 2)

    # Smart text handling
    if capex < 0:
        capex_text = "with a net gain upfront"
    elif capex == 0:
        capex_text = "with no upfront investment required"
    else:
        capex_text = f"with a typical CAPEX of £{int(capex):,}"

    if roi < 0:
        roi_text = "ROI has been negative in some cases — often due to poor alignment, which we help address"
    elif roi == 0:
        roi_text = "returns have varied — some businesses break even, others save significantly"
    else:
        roi_text = f"typical ROI of {roi}%"

    if payback <= 0:
        payback_text = "with immediate or near-instant payback"
    else:
        payback_text = f"with payback in ~{payback} months"

    if financial_savings < 0:
        savings_text = "Some companies have reported negative returns — highlighting the need for better initiative matching"
    elif financial_savings < 500:
        savings_text = f"modest savings of ~£{int(financial_savings):,}/yr are typical"
    else:
        savings_text = f"typical annual savings of ~£{int(financial_savings):,}"

    # Industry & Geo Benchmarks
    ind_emissions = round(ind_stats['emissions_t_CO2e'], 1)
    geo_emissions = round(geo_stats['emissions_t_CO2e'], 1)
    ind_emp = int(ind_stats['employees'])
    geo_emp = int(geo_stats['employees'])
    ind_turnover = int(ind_stats['turnover'])
    geo_turnover = int(geo_stats['turnover'])

    benchmarks_text = f"""
Across {industry_map[industry]}, SMBs average ~{ind_emp} employees and ~£{ind_turnover:,} in annual turnover.
In {geography}, similar firms emit ~{geo_emissions:,} tCO₂e annually and have ~{geo_emp} staff on average.
This means your profile aligns closely with the peers we've analyzed.
""".strip()

    # Top initiatives (up to 5)
    if len(possible_initiatives) < 5:
        top_inits = possible_initiatives.sort_values(by='avg_savings', ascending=False)
    else:
        top_inits = possible_initiatives.sort_values(by='avg_savings', ascending=False).head(5)

    top_init_lines = "\n".join([
        f"✅ {row['initiative_name']}: ~£{int(row['avg_savings']):,}/yr, ~{round(row['avg_carbon_savings'], 1)} tCO₂e saved, ROI ~{round(row['avg_financial_roi'], 1)}%, payback ~{'immediate' if row['avg_payback'] <= 0 else str(round(row['avg_payback'], 1)) + ' months'}. Already adopted by {'one' if int(row['num_companies']) == 1 else int(row['num_companies'])} SMB{'s' if int(row['num_companies']) != 1 else ''} in your industry."
        for _, row in top_inits.iterrows()
    ])

    # Final email text
    email = f"""
Subject: See how {industry_map[industry]} firms in {geography} save £{int(financial_savings):,}/yr & cut emissions

Hi {company_name},

Did you know that similar businesses in {geography} within the {industry_map[industry]} industry are achieving substantial results by acting on their emissions?

Here’s what we’ve learned from our analysis of hundreds of SMBs like yours in {geography}:

- 🌍 Average carbon reduction: ~{carbon_savings:,} tCO₂e
- 💸 {savings_text}
- ⚙️ {roi_text}, {payback_text}
- 🧾 {capex_text}
- 📊 Opportunity score: {score} (vs national average)

{benchmarks_text}

🔍 Top actions businesses like yours are implementing:
{top_init_lines}

These aren’t theoretical models — they're based on companies just like yours. The difference? They're already seeing results. We can help you do the same.

Would you be open to a 20-minute chat this week?

Best regards,  
Leonard Onyiriuba  
XCD Climate AI  
lonyiriuba@xcdclimateai.com
    """.strip()

    return email


# # ---------------------------------------
# # Example use case for function

# smb_profile = {
#     'company_name': 'A&E Limited',
#     'industry_name': 'F: CONSTRUCTION',
#     'geography_name': 'Vale of White Horse',
#     'employees': 50,
#     'turnover': 3_000_000,  # GBP
#     'emissions_t_CO2e': 100
# }
# email = generate_cold_email(profile = smb_profile)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate a cold outreach email for a given SMB profile.")
    parser.add_argument("--company_name", type=str, required=True)
    parser.add_argument("--industry_name", type=str, required=True)
    parser.add_argument("--geography_name", type=str, required=True)
    parser.add_argument("--employees", type=int, required=True)
    parser.add_argument("--turnover", type=float, required=True)
    parser.add_argument("--emissions_t_CO2e", type=float, required=True)
    parser.add_argument("--output", type=str, default="output_email.txt", help="Path to output file")

    args = parser.parse_args()

    profile = {
        "company_name": args.company_name,
        "industry_name": args.industry_name,
        "geography_name": args.geography_name,
        "employees": args.employees,
        "turnover": args.turnover,
        "emissions_t_CO2e": args.emissions_t_CO2e
    }

    email_text = generate_cold_email(profile)

    # Print to console
    print("\n" + "="*80)
    print("GENERATED EMAIL\n")
    print(email_text)
    print("="*80 + "\n")

    # Write to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(email_text)
    print(f"✅ Email written to {args.output}")

