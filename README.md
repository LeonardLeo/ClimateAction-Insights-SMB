# ClimateAction-Insights-SMB

**Author**: Leonard Onyiriuba  
**Last Updated**: May 2025

## 📘 Overview

**ClimateAction-Insights-SMB** is a data analytics and predictive modeling project designed to help Small and Medium-sized Businesses (SMBs) make informed, data-driven decisions around carbon reduction initiatives. The solution combines exploratory data analysis, predictive modeling, and a Power BI dashboard to surface insights that can support strategic sustainability planning.

---

## 📁 Project Structure

```text
ClimateAction-Insights-SMB/
├── data/                            # Raw and preprocessed data files
│   ├── company_initiatives.parquet
│   └── deidentified_company_initiatives.xlsx
│
├── eda/                             # Exploratory Data Analysis
│   ├── correlation_summary/         # Correlation matrices and insights
│   ├── eda_clean_data/              # Cleaned datasets
│   ├── eda_encoded_data/            # Encoded versions of datasets
│   ├── eda_model_data/              # Final datasets used in modeling
│   ├── eda_processed_data/          # Intermediate processing
│   └── eda_viz/                     # EDA visualizations
│
├── predictive_modeling/            # Regression models for prediction
│   ├── Predicting_Initiative_Carbon_Savings_t_CO2e/
│   │   └── results/                 # Models and results
│   └── Predicting_Initiative_Financial_Savings/
│       └── results/                 # Models and results
│
├── smb_data_analysis/              # Data prep for dashboard
├── dashboard.pbix                  # Power BI dashboard file
├── utils.py                        # Utility functions
├── notes.txt                       # Research notes and scratchpad
├── README.md                       # Project documentation
```


---

## 💡 Key Features

- 📊 **Exploratory Data Analysis (EDA)**  
  Comprehensive correlation, distribution, and feature analysis to understand company behavior and initiative impacts.

- 🤖 **Predictive Modeling**  
  Two regression models:
  - Predicting initiative carbon savings (`t_CO2e`)
  - Predicting financial savings from initiatives (`GBP`)

- 📉 **Feature Selection & Transformation**  
  Includes statistical feature selection (`SelectFpr`, `SelectKBest`), log transformation, and standardization.

- 📈 **Performance Metrics**  
  Reports `R²`, `MAE`, `MSE`, `RMSE`, and cross-validation scores to ensure model robustness.

- 📊 **Interactive Dashboard (Power BI)**  
  Visual analytics to explore relationships between company profiles and carbon-related initiative outcomes.

---

## 📌 Power BI Dashboard

🔗 [PowerBI Dashboard](https://app.powerbi.com/view?r=eyJrIjoiZGYxMThkYmEtOWY4NS00NWEyLWJjM2EtM2NhN2Y1YzJjZDNlIiwidCI6IjY2NjYxMWFjLTE1NjktNDhjYy1iYjg5LWY2MjZkY2JmMjkxMSJ9&embedImagePlaceholder=true&pageName=0de1aa8aaa5d2237e1bb)

---

## 🧰 Requirements

> Although no `requirements.txt` was used, the core libraries include:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `openpyxl`

You can install them manually or use your preferred environment manager.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ClimateAction-Insights-SMB.git
   cd ClimateAction-Insights-SMB

2. Open dashboard.pbix in Power BI Desktop to explore the interactive dashboard.

