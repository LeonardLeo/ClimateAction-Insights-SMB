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
from ydata_profiling import ProfileReport
from utils import eda



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
dataset = pd.read_parquet("data/company_initiatives.parquet")
columns = dataset.columns

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
EDA_data = eda(dataset)

# =============================================================================
# EDA Visualization
# =============================================================================
profile = ProfileReport(dataset, title="SMB Climate EDA", explorative=True)
profile.to_file("eda_viz/climate_report.html")

# =============================================================================
# Data Cleaning & Preprocessing
# =============================================================================
