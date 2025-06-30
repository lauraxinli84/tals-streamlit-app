import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from functools import partial
from sklearn.linear_model import LinearRegression
from contextlib import contextmanager
import pickle
from pathlib import Path
import os
import re
import joblib
import shap
from streamlit_shap import st_shap
from preprocessing import preprocess_client_data, interpret_risk_score
import gspread
from google.oauth2.service_account import Credentials
import json
import requests
import hashlib

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password():
    """Returns True if the user has entered the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        # Define valid users and their hashed passwords
        # You can add more users here
        valid_users = {
            'admin': '5f358e17be89fc2c070a2066e6ce8e4dc4eb96cab770c8949ef808ec053cee80',
            'laet': 'b28175c8d274ea5c7cfb0d9ef6a6ea9ce58ea8615aa15b12e268e6e53bd8cd76',
            'wtls': '20b0f7d1b03c42781fba82a79fce61559b771456d7c234e86d2791b5d3a36605',
            'mals': '52768119e3adca086712dabab4328eb92319d826fcf059f5d7998213ad47c024',
            'las': '3acb16573f9788cd955704c9519df5b9c5aac37e4c9b4099e55ae257fe6f4757',
            'tals': '4f4cea335e3bf099b6d62d2de3f00b7aeca7886c5d8c6910de5dfe4e232ea0a6'
        }
    
        if username in valid_users and hash_password(password) == valid_users[username]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
            del st.session_state["username"]  # Don't store the username
        else:
            st.session_state["password_correct"] = False

    # Return True if password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown("# 🔒 TALS Data Explorer - Login Required")
    st.markdown("Please enter your credentials to access the application.")
    
    with st.form("login_form"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        submit_button = st.form_submit_button("Login", on_click=password_entered)
    
    if "password_correct" in st.session_state:
        if not st.session_state["password_correct"]:
            st.error("😞 Username or password incorrect")
    
    st.markdown("---")
    st.markdown("*Contact your administrator if you need access credentials.*")
    
    return False

# Call the authentication function - ADD THIS RIGHT AFTER YOUR IMPORTS
if not check_password():
    st.stop()  # Do not continue if check_password is not True

# Set page config
st.set_page_config(page_title="TALS Data Explorer", layout="wide")

# Initialize session state defaults once at app startup
st.session_state.setdefault('upload_stage', 'initial')
st.session_state.setdefault('processed_data', None)
st.session_state.setdefault('upload_success', False)
st.session_state.setdefault('saving_in_progress', False)

# Standardization mappings
@st.cache_data
def get_standard_mappings():
    # Column mapping
    column_mapping = {
        'Client ID': 'client_id',
        'Matter/Case ID': 'case_id',
        'Case # ID': 'case_id',
        
        'Date Opened': 'date_opened',
        'Opened': 'date_opened',
        'Date Closed': 'date_closed',
        'Closed': 'date_closed',
        'Number of Days Open': 'days_open',
        '# Days Open': 'days_open',
        
        'Percentage of Poverty': 'poverty_pct',
        'Poverty %': 'poverty_pct',
        'Adjusted Percentage of Poverty': 'adj_poverty_pct',
        'Adj. Poverty %': 'adj_poverty_pct',
        'Income Eligible': 'income_eligible',
        'Financial Eligibility Override Reason': 'income_override_reason',
        'Financial Override Reason': 'income_override_reason',
        'Income Waiver Request Status': 'income_waiver_status',
        'Asset Eligible': 'asset_eligible',
        'Asset Override Reason': 'asset_override_reason',
        'Asset Waiver Request Status': 'asset_waiver_status',
        
        'Gender': 'gender',
        'Race': 'race',
        'HUD 9902 Ethnicity': 'ethnicity',
        'Ethnicity': 'ethnicity',
        'Age at Intake': 'age_intake',
        'Intake Age': 'age_intake',
        'Disabled': 'disabled',
        'Living Arrangement': 'living_arrangement',
        'Veteran': 'veteran',
        'Language': 'language',
        'Identifies as LGBT?': 'lgbt',
        'LGBTQ': 'lgbt',
        'Citizenship Status': 'citizenship',
        'Citizenship': 'citizenship',
        
        'Total Household Size': 'household_total',
        'Total Household': 'household_total',
        'Number of People 18 and Over': 'household_adults',
        'People > 18': 'household_adults',
        'Number of People under 18': 'household_children',
        'People < 18': 'household_children',
        
        'County of Residence': 'county_residence',
        'Zip Code': 'zip_code',
        'County of Dispute': 'county_dispute',
        
        'Legal Problem Code': 'legal_problem_code',
        'Close Reason': 'close_reason',
        'Funding Source': 'funding_source',
        'PAI Case?': 'pai_case',
        'PAI Case': 'pai_case',
        
        'How did Applicant hear about LAET?': 'referral_source',
        'How did Applicant hear about LAS?': 'referral_source',
        'How did Applicant hear about WTLS?': 'referral_source',
        'Outcome': 'outcome',
        'Outcome Value Category': 'outcome_category',
        'Outcome value category': 'outcome_category',
        'Outcome Amount': 'outcome_amount',
        'Total Time For Case': 'case_time',
        'Domestic Violence Present': 'domestic_violence',
        'Is the caller a victim of domestic violence?': 'domestic_violence'
    }
    
    # Race mapping
    race_mapping = {
        # White categories
        'White': 'White',
        'White (Not Hispanic)': 'White',
        'Caucasian/White': 'White',
        'White - Not Hispanic': 'White',
        
        # Black categories
        'Black': 'Black',
        'Black (Not Hispanic)': 'Black',
        'African American/Black': 'Black',
        'Black or African American': 'Black',
        'Black - Not Hispanic': 'Black',
        
        # Native American categories
        'Native American': 'Native American',
        'Native American or Alaska Native': 'Native American',
        'American Indian or Alaska Native and White': 'Native American',
        
        # Asian and Pacific Islander categories
        'Asian': 'Asian/Pacific Islander',
        'Asian or Pacific Islander': 'Asian/Pacific Islander',
        'Native Hawaiian or Other Pacific Islander': 'Asian/Pacific Islander',
        
        # Hispanic
        'Hispanic': 'Hispanic',
        
        # Multiracial categories
        'Mulitracial': 'Multiracial',
        'Black or African American and White': 'Multiracial',
        'Asian and White': 'Multiracial',
        
        # Other categories
        'Other': 'Other/Unknown',
        'Other/Unknown': 'Other/Unknown',
        'No Response': 'Other/Unknown',
        None: 'Other/Unknown'
    }
    
    # legal problem code mapping
    legal_problem_mapping = {
    # Consumer/Finance (01-09)
    r'(?i)^\s*0?1\s*[-: ]?\s*.*?(?:bankrupt|debtor)': '01 Bankruptcy/Debtor Relief',
    r'(?i)^\s*0?2\s*[-: ]?\s*.*?(?:collect|repo|def|garn)': '02 Collection (including Repo/Def/Garnish)',
    r'(?i)^\s*0?3\s*[-: ]?\s*.*?(?:contract|warrant)': '03 Contracts/Warranties',
    r'(?i)^\s*0?4\s*[-: ]?\s*.*?(?:collect.*?practi|creditor|harass)': '04 Collection Practices/Creditor Harassment',
    r'(?i)^\s*0?5\s*[-: ]?\s*.*?(?:predat.*?lend|lend.*?practice)(?!.*mortgage)': '05 Predatory Lending Practices (not mortgages)',
    r'(?i)^\s*0?6\s*[-: ]?\s*.*?(?:loan|install.*?purch)': '06 Loans/Installment Purch.',
    r'(?i)^\s*0?7\s*[-: ]?\s*.*?(?:public.*?util|utilit)': '07 Public Utilities',
    r'(?i)^\s*0?8\s*[-: ]?\s*.*?(?:unfair|decept).*?(?:sales|practice)(?!.*real.*prop)': '08 Unfair and Deceptive Sales and Practices (not real property)',
    r'(?i)^\s*0?9\s*[-: ]?\s*.*?(?:consumer|finance)': '09 Other Consumer/Finance',

    # Education (12-19)
    r'(?i)^\s*1?2\s*[-: ]?\s*.*?(?:discipl|expul|suspen)': '12 Discipline (including expulsion and suspension)',
    r'(?i)^\s*1?3\s*[-: ]?\s*.*?(?:special.*?ed|learn.*?disab)': '13 Special Education/Learning Disabilities',
    r'(?i)^\s*1?4\s*[-: ]?\s*.*?(?:access|biling|resid|test)': '14 Access (Including Bilingual, Residency, Testing)',
    r'(?i)^\s*1?6\s*[-: ]?\s*.*?(?:student|financ.*?aid)': '16 Student Financial Aid',
    r'(?i)^\s*1?9\s*[-: ]?\s*.*?(?:educ)': '19 Other Education',

    # Employment (21-29)
    r'(?i)^\s*2?1\s*[-: ]?\s*.*?(?:employ.*?discrim)': '21 Employment Discrimination',
    r'(?i)^\s*2?2\s*[-: ]?\s*.*?(?:wage|flsa)': '22 Wage Claim and other FLSA Issues',
    r'(?i)^\s*2?3\s*[-: ]?\s*.*?(?:eitc|earn.*?income.*?tax)': '23 EITC (Earned Income Tax Credit)',
    r'(?i)^\s*2?4\s*[-: ]?\s*.*?(?:tax)(?!.*eitc)': '24 Taxes (not EITC)',
    r'(?i)^\s*2?5\s*[-: ]?\s*.*?(?:employ.*?right)': '25 Employee Rights',
    r'(?i)^\s*2?9\s*[-: ]?\s*.*?(?:employ|ceta)': '29 Other Employment',

    # Family (30-39)
    r'(?i)^\s*3?0\s*[-: ]?\s*.*?(?:adopt)': '30 Adoption',
    r'(?i)^\s*3?1\s*[-: ]?\s*.*?(?:custody|visit)': '31 Custody/Visitation',
    r'(?i)^\s*3?2\s*[-: ]?\s*.*?(?:divorce|sep|annul)': '32 Divorce/Sep./Annul.',
    r'(?i)^\s*3?3\s*[-: ]?\s*.*?(?:adult.*?guard|conserv)': '33 Adult Guardianship/Conserv.',
    r'(?i)^\s*3?4\s*[-: ]?\s*.*?(?:name.*?change)': '34 Name Change',
    r'(?i)^\s*3?5\s*[-: ]?\s*.*?(?:parent.*?right.*?term)': '35 Parental Rights Termin.',
    r'(?i)^\s*3?6\s*[-: ]?\s*.*?(?:patern)': '36 Paternity',
    r'(?i)^\s*3?7\s*[-: ]?\s*.*?(?:dom.*?abuse)': '37 Domestic Abuse',
    r'(?i)^\s*3?8\s*[-: ]?\s*.*?(?:support)': '38 Support',
    r'(?i)^\s*3?9\s*[-: ]?\s*.*?(?:family)': '39 Other Family',

    # Juvenile (41-49)
    r'(?i)^\s*4?1\s*[-: ]?\s*.*?(?:delinq)': '41 Delinquent',
    r'(?i)^\s*4?2\s*[-: ]?\s*.*?(?:neglect|abuse|depend)': '42 Neglected/Abused/Depend.',
    r'(?i)^\s*4?3\s*[-: ]?\s*.*?(?:emancip)': '43 Emancipation',
    r'(?i)^\s*4?4\s*[-: ]?\s*.*?(?:minor.*?guard|conserv)': '44 Minor Guardian/Conservatorship',
    r'(?i)^\s*4?9\s*[-: ]?\s*.*?(?:juvenile)': '49 Other Juvenile',

    # Health (51-59)
    r'(?i)^\s*5?1\s*[-: ]?\s*.*?(?:medicaid|tenncare)': '51 Medicaid',
    r'(?i)^\s*5?2\s*[-: ]?\s*.*?(?:medicare)': '52 Medicare',
    r'(?i)^\s*5?3\s*[-: ]?\s*.*?(?:govern.*?child.*?health|insur.*?program)': "53 Government Children's Health Insurance Programs",
    r'(?i)^\s*5?4\s*[-: ]?\s*.*?(?:home.*?comm.*?base|care)': '54 Home and Community Based Care',
    r'(?i)^\s*5?5\s*[-: ]?\s*.*?(?:private.*?health.*?insur)': '55 Private Health Insurance',
    r'(?i)^\s*5?6\s*[-: ]?\s*.*?(?:long.*?term.*?health|care.*?facil)': '56 Long Term Health Care Facilities',
    r'(?i)^\s*5?7\s*[-: ]?\s*.*?(?:state.*?local.*?health)': '57 State and Local Health',
    r'(?i)^\s*5?9\s*[-: ]?\s*.*?(?:health)': '59 Other Health',

    # Housing (61-69)
    r'(?i)^\s*6?1\s*[-: ]?\s*.*?(?:fed.*?subsid.*?hous|subsid.*?hous)': '61 Fed. Subsidized Housing',
    r'(?i)^\s*6?2\s*[-: ]?\s*.*?(?:homeown|real.*?prop)(?!.*foreclos)': '62 Homeownership/Real Prop. (not foreclosure)',
    r'(?i)^\s*6?3\s*[-: ]?\s*.*?(?:private.*?land|tenant)': '63 Private Landlord/Tenant',
    r'(?i)^\s*6?4\s*[-: ]?\s*.*?(?:public.*?hous)': '64 Public Housing',
    r'(?i)^\s*6?5\s*[-: ]?\s*.*?(?:mobile.*?home)': '65 Mobile Homes',
    r'(?i)^\s*6?6\s*[-: ]?\s*.*?(?:hous.*?discrim)': '66 Housing Discrimination',
    r'(?i)^\s*6?7\s*[-: ]?\s*.*?(?:mortgage.*?forecl)(?!.*predat)': '67 Mortgage Foreclosures (not predatory Lending/practices)',
    r'(?i)^\s*6?8\s*[-: ]?\s*.*?(?:mortgage.*?predat|predat.*?lend)': '68 Mortgage Predatory Lending/Practices',
    r'(?i)^\s*6?9\s*[-: ]?\s*.*?(?:hous)': '69 Other Housing',

    # Income Maintenance (71-79)
    r'(?i)^\s*7?1\s*[-: ]?\s*.*?(?:tanf|famil.*?first)': '71 TANF',
    r'(?i)^\s*7?2\s*[-: ]?\s*.*?(?:social.*?secur)(?!.*ssdi)': '72 Social Security (not SSDI)',
    r'(?i)^\s*7?3\s*[-: ]?\s*.*?(?:food.*?stamp)': '73 Food Stamps',
    r'(?i)^\s*7?4\s*[-: ]?\s*.*?(?:ssdi)': '74 SSDI',
    r'(?i)^\s*7?5\s*[-: ]?\s*.*?(?:ssi)': '75 SSI',
    r'(?i)^\s*7?6\s*[-: ]?\s*.*?(?:unemploy.*?comp)': '76 Unemployment Compensation',
    r'(?i)^\s*7?7\s*[-: ]?\s*.*?(?:veteran.*?bene)': '77 Veterans Benefits',
    r'(?i)^\s*7?8\s*[-: ]?\s*.*?(?:state.*?local.*?income)': '78 State and Local Income Maintenance',
    r'(?i)^\s*7?9\s*[-: ]?\s*.*?(?:income|mainten)': '79 Other Income Maintenance',

    # Rights and Other (81-89)
    r'(?i)^\s*8?1\s*[-: ]?\s*.*?(?:immigr|natural)': '81 Immigration/Naturalization',
    r'(?i)^\s*8?2\s*[-: ]?\s*.*?(?:mental.*?health)': '82 Mental Health',
    r'(?i)^\s*8?4\s*[-: ]?\s*.*?(?:disab.*?right)': '84 Disability Rights',
    r'(?i)^\s*8?5\s*[-: ]?\s*.*?(?:civil.*?right)': '85 Civil Rights',
    r'(?i)^\s*8?6\s*[-: ]?\s*.*?(?:human.*?traffic)': '86 Human Trafficking',
    r'(?i)^\s*8?7\s*[-: ]?\s*.*?(?:expung)': '87 Expungement',
    r'(?i)^\s*8?9\s*[-: ]?\s*.*?(?:other.*?individ.*?right|individual.*?right)': '89 Other Individual Rights',

    # Miscellaneous (93-99)
    r'(?i)^\s*9?3\s*[-: ]?\s*.*?(?:licens)': '93 Licenses (Auto and Other)',
    r'(?i)^\s*9?4\s*[-: ]?\s*.*?(?:tort)': '94 Torts',
    r'(?i)^\s*9?5\s*[-: ]?\s*.*?(?:will|estat)': '95 Wills/Estates',
    r'(?i)^\s*9?6\s*[-: ]?\s*.*?(?:advan.*?direct|power.*?attorney)': '96 Advance Directives/Powers of Attorney',
    r'(?i)^\s*9?7\s*[-: ]?\s*.*?(?:munic.*?legal)': '97 Municipal Legal Needs',
    r'(?i)^\s*9?9\s*[-: ]?\s*.*?(?:misc|other)': '99 Other Miscellaneous'
}
    return column_mapping, race_mapping, legal_problem_mapping

# Function to apply regex-based legal problem mapping
def map_legal_problem_with_regex(problem_code, legal_problem_patterns):
    if pd.isna(problem_code):
        return None
    
    # Convert to string to ensure compatibility
    problem_str = str(problem_code).strip()

    direct_mappings = {
        '01 Bankruptcy/Debtor Relief': '01 Bankruptcy/Debtor Relief',
        '02 Collection (including Repo/Def/Garnish)': '02 Collection (including Repo/Def/Garnish)',
        '02 Collect/Repo/Def/Garnsh': '02 Collection (including Repo/Def/Garnish)',
        '02 - Collections (Repo, Def., Garn)': '02 Collection (including Repo/Def/Garnish)',
        '03 Contracts / Warranties': '03 Contracts/Warranties',
        '03 Contract/Warranties': '03 Contracts/Warranties',
        '04 Collection Practices/Creditor Harassment': '04 Collection Practices/Creditor Harassment',
        '04 Collection Practices / Creditor Harassment': '04 Collection Practices/Creditor Harassment',
        '05 Predatory Lending Practices (not mortgages)': '05 Predatory Lending Practices (not mortgages)',
        '06 Loans/Installment Purch.': '06 Loans/Installment Purch.',
        '06 Loans/Installment Purchases (Not Collections)': '06 Loans/Installment Purch.',
        '07 Public Utilities': '07 Public Utilities',
        '08 Unfair and Deceptive Sales and Practices (not real property)': '08 Unfair and Deceptive Sales and Practices (not real property)',
        '08 Unfair and Deceptive Sales Practices (Not Real Property)': '08 Unfair and Deceptive Sales and Practices (not real property)',
        '09 Other Consumer/Finance': '09 Other Consumer/Finance',
        '09 Other Consumer / Finance.': '09 Other Consumer/Finance',

        # Education (12-19)
        '12 Discipline (including expulsion and suspension)': '12 Discipline (including expulsion and suspension)',
        '12 Discipline (Including Expulsion and Suspension)': '12 Discipline (including expulsion and suspension)',
        '13 Special Education/Learning Disabilities': '13 Special Education/Learning Disabilities',
        '14 Access (Including Bilingual, Residency, Testing)': '14 Access (Including Bilingual, Residency, Testing)',
        '16 Student Financial Aid': '16 Student Financial Aid',
        '19 Other Education': '19 Other Education',

        # Employment (21-29)
        '21 Employment Discrimination': '21 Employment Discrimination',
        '22 Wage Claim and other FLSA Issues': '22 Wage Claim and other FLSA Issues',
        '22 Wage Claims and Other FLSA Issues': '22 Wage Claim and other FLSA Issues',
        '23 EITC (Earned Income Tax Credit)': '23 EITC (Earned Income Tax Credit)',
        '24 Taxes (not EITC)': '24 Taxes (not EITC)',
        '24 Taxes (Not EITC)': '24 Taxes (not EITC)',
        '25 Employee Rights': '25 Employee Rights',
        '29 Other Employment & Ceta': '29 Other Employment',
        '29 Other Employment': '29 Other Employment',

        # Family (30-39)
        '30 Adoption': '30 Adoption',
        '31 Custody/Visitation': '31 Custody/Visitation',
        '31 Custody / Visitation': '31 Custody/Visitation',
        '32 Divorce/Sep./Annul.': '32 Divorce/Sep./Annul.',
        '32 Divorce / Sep. / Annul.': '32 Divorce/Sep./Annul.',
        '33 Adult Guardianship / Conserv.': '33 Adult Guardianship/Conserv.',
        '33 Adult Guardianship / Conservatorship': '33 Adult Guardianship/Conserv.',
        '34 Name Change': '34 Name Change',
        '35 Parental Rights Termin.': '35 Parental Rights Termin.',
        '35 Parental Rights Termination': '35 Parental Rights Termin.',
        '36 Paternity': '36 Paternity',
        '37 Domestic Abuse': '37 Domestic Abuse',
        '37 - Domestic Abuse': '37 Domestic Abuse',
        '38 Support': '38 Support',
        '39 Other Family': '39 Other Family',

        # Juvenile (41-49)
        '41 Delinquent': '41 Delinquent',
        '42 Neglected/Abused/Depend.': '42 Neglected/Abused/Depend.',
        '42 Neglected/Abused/Dependent': '42 Neglected/Abused/Depend.',
        '43 Emancipation': '43 Emancipation',
        '44 Minor Guardian/Conservatorship': '44 Minor Guardian/Conservatorship',
        '44 Minor Guardianship / Conservatorship': '44 Minor Guardian/Conservatorship',
        '49 Other Juvenile': '49 Other Juvenile',

        # Health (51-59)
        '51 Medicaid': '51 Medicaid',
        '51 - Medicaid (Tenncare)': '51 Medicaid',
        '52 Medicare': '52 Medicare',
        "53 Goverment Children's Health Insurance Programs": "53 Government Children's Health Insurance Programs",
        '54 Home and Community Based Care': '54 Home and Community Based Care',
        '55 Private Health Insurance': '55 Private Health Insurance',
        '56 Long Term Health Care Facilities': '56 Long Term Health Care Facilities',
        '57 State and Local Health': '57 State and Local Health',
        '59 Other Health': '59 Other Health',

        # Housing (61-69)
        '61 Fed. Subsidized Housing': '61 Fed. Subsidized Housing',
        '61 Federally Subsidized Housing': '61 Fed. Subsidized Housing',
        '61 - Federally Subsidized Housing': '61 Fed. Subsidized Housing',
        '62 Homeownership/Real Prop. (not foreclosure)': '62 Homeownership/Real Prop. (not foreclosure)',
        '62 Homeownership/Real Property (Not Foreclosure)': '62 Homeownership/Real Prop. (not foreclosure)',
        '63 Private Landlord / Tenant': '63 Private Landlord/Tenant',
        '63 Private Landlord/Tenant': '63 Private Landlord/Tenant',
        '63 - Private Landlord/Tenant': '63 Private Landlord/Tenant',
        '64 Public Housing': '64 Public Housing',
        '65 Mobile Homes': '65 Mobile Homes',
        '66 Housing Discrimination': '66 Housing Discrimination',
        '67 Mortgage Foreclosures (not predatory Lending/practices)': '67 Mortgage Foreclosures (not predatory Lending/practices)',
        '67 Mortgage Foreclosures (Not Predatory Lending/Practices)': '67 Mortgage Foreclosures (not predatory Lending/practices)',
        '68 Mortgage Predatory Lending/Practices': '68 Mortgage Predatory Lending/Practices',
        '69 Other Housing': '69 Other Housing',

        # Income Maintenance (71-79)
        '71 TANF': '71 TANF',
        '71 - TANF (Families First)': '71 TANF',
        '72 Social Security (not SSDI)': '72 Social Security (not SSDI)',
        '72 Social Security (Not SSDI)': '72 Social Security (not SSDI)',
        '73 Food Stamps': '73 Food Stamps',
        '73 Food Stamps / Commodities': '73 Food Stamps',
        '74 SSDI': '74 SSDI',
        '75 SSI': '75 SSI',
        '76 Unemployment Compensation': '76 Unemployment Compensation',
        '77 Veterans Benefits': '77 Veterans Benefits',
        '78 State and Local Income Maintenance': '78 State and Local Income Maintenance',
        '79 Other Income Maintenance': '79 Other Income Maintenance',
        '79 Other Income Maintenence': '79 Other Income Maintenance',

        # Rights and Other (81-89)
        '81 Immigration/Naturalization': '81 Immigration/Naturalization',
        '81 Immigration / Naturalization': '81 Immigration/Naturalization',
        '82 Mental Health': '82 Mental Health',
        '84 Disability Rights': '84 Disability Rights',
        '85 Civil Rights': '85 Civil Rights',
        '86 Human Trafficking': '86 Human Trafficking',
        '87 Expungement': '87 Expungement',
        '87 - Expungement': '87 Expungement',
        '87 Criminal Record Expungement': '87 Expungement',
        '89 Other Individual Rights': '89 Other Individual Rights',

        # Miscellaneous (93-99)
        '93 Licenses (Auto and Other)': '93 Licenses (Auto and Other)',
        '93 Licenses (Drivers, Occupational, and Others)': '93 Licenses (Auto and Other)',
        '94 Torts': '94 Torts',
        '95 Wills / Estates': '95 Wills/Estates',
        '95 Wills and Estates': '95 Wills/Estates',
        '96 Advance Directives/Powers of Attorney': '96 Advance Directives/Powers of Attorney',
        '96 Advanced Directives/Powers of Attorney': '96 Advance Directives/Powers of Attorney',
        '97 Municipal Legal Needs': '97 Municipal Legal Needs',
        '99 Other Miscellaneous': '99 Other Miscellaneous'
    }
    
    # First try direct mapping for efficiency
    if problem_str in direct_mappings:
        return direct_mappings[problem_str]
    
    # Then try regex patterns
    for pattern, standardized_code in legal_problem_patterns.items():
        if re.search(pattern, problem_str):
            return standardized_code
    
    # Return original if no match found
    return problem_code

# Standardization function
def standardize_new_data(df, upload_source):  
    column_mapping, race_mapping, legal_problem_mapping = get_standard_mappings()
    
    # First standardize the column names
    df = df.rename(columns=column_mapping)
    
    # Define the expected column order 
    column_order = [
        # Identifying Information
        'client_id',
        'case_id',
        'source',  
        
        # Dates and Duration
        'date_opened',
        'date_closed',
        'days_open',
        'case_time',
        
        # Financial Eligibility
        'poverty_pct',
        'adj_poverty_pct',
        'income_eligible',
        'income_override_reason',
        'income_waiver_status',
        'asset_eligible',
        'asset_override_reason',
        'asset_waiver_status',
        
        # Demographics
        'age_intake',
        'gender',
        'race',
        'ethnicity',
        'disabled',
        'veteran',
        'language',
        'lgbt',
        'citizenship',
        
        # Household Information
        'household_total',
        'household_adults',
        'household_children',
        'living_arrangement',
        
        # Location
        'county_residence',
        'zip_code',
        'county_dispute',
        
        # Case Details
        'legal_problem_code',
        'funding_source',
        'pai_case',
        'referral_source',
        'domestic_violence',
        
        # Outcome Information
        'close_reason',
        'outcome_category',
        'outcome_amount',
        'outcome'
    ]

    # Standardize race categories if present
    if 'race' in df.columns:
        df['race'] = df['race'].replace(race_mapping)

    # Standardize legal problem codes if present
    if 'legal_problem_code' in df.columns:
        df['legal_problem_code'] = df['legal_problem_code'].apply(lambda x: map_legal_problem_with_regex(x, legal_problem_mapping))

    # Clean and normalize 'domestic_violence'
    if 'domestic_violence' in df.columns:
        # Strip leading/trailing whitespace and ensure string type
        df['domestic_violence'] = df['domestic_violence'].astype(str).str.strip()

        # Replace known valid entries; everything else becomes NaN
        df['domestic_violence'] = df['domestic_violence'].apply(
            lambda x: x if x in ['Yes', 'No'] else np.nan
        )

    # Normalize income_eligible
    if 'income_eligible' in df.columns:
        df['income_eligible'] = df['income_eligible'].astype(str).str.strip().str.capitalize()
        df['income_eligible'] = df['income_eligible'].apply(
            lambda x: x if x in ['Yes', 'No'] else np.nan
        )

    # Normalize asset_eligible
    if 'asset_eligible' in df.columns:
        df['asset_eligible'] = df['asset_eligible'].astype(str).str.strip().str.capitalize()
        df['asset_eligible'] = df['asset_eligible'].apply(
            lambda x: x if x in ['Yes', 'No'] else np.nan
        )

    # Add missing columns with nan
    for col in column_order:
        if col not in df.columns:
            if col == 'source':
                df[col] = upload_source  # Use the provided organization source
            else:
                df[col] = np.nan
    
    # Convert household columns to numeric
    household_cols = ['household_total', 'household_adults', 'household_children']
    for col in household_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['poverty_pct', 'adj_poverty_pct', 'age_intake', 'outcome_amount', 'case_time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle date columns
    date_cols = ['date_opened', 'date_closed']
    for col in date_cols:
        if col in df.columns:
            # Convert to datetime and normalize to remove time component
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
        
    # Ensure all columns are in the same order
    df = df[column_order]
    
    return df


def get_google_credentials():
    """
    Get Google credentials from Streamlit secrets
    """
    try:
        if hasattr(st, 'secrets') and 'google_credentials' in st.secrets:
            return dict(st.secrets.google_credentials)
        else:
            st.error("Google credentials not found in Streamlit secrets.")
            return None
    except Exception as e:
        st.error(f"Error loading credentials: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    """
    Load data from Google Drive using service account credentials
    """
    try:
        # Get credentials from Streamlit secrets
        creds_dict = get_google_credentials()
        if creds_dict is None:
            st.error("Cannot load data: Missing credentials")
            st.stop()
        
        # Set up credentials and authorize
        scopes = ['https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        # Your Google Drive file ID
        FILE_ID = "1IaVYJsgyqno73-O-s0TMD1AoQHRdkpsUz-LonB1CbF4"
        
        # Open the file and get the first worksheet
        file = gc.open_by_key(FILE_ID)
        worksheet = file.get_worksheet(0)
        
        # Get all values and convert to DataFrame
        data = worksheet.get_all_values()
        headers = data[0]
        rows = data[1:]
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Convert date columns 
        date_columns = ['date_opened', 'date_closed']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', cache=False).dt.normalize()
        
        # Convert numeric columns (EXPANDED LIST)
        numeric_columns = [
            'poverty_pct', 'adj_poverty_pct', 'age_intake', 'outcome_amount', 'case_time',
            'household_total', 'household_adults', 'household_children', 'days_open'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Optimize categorical columns
        categorical_columns = [
            'source', 'gender', 'race', 'ethnicity', 'county_residence',
            'county_dispute', 'legal_problem_code', 'close_reason',
            'funding_source', 'outcome_category'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col])
            
        return df
        
    except Exception as e:
        st.error(f"An error occurred loading data from Google Drive: {str(e)}")
        st.stop()

# functions for data upload 
def save_to_google_drive(combined_df):
    """
    Save updated dataframe back to Google Drive
    """
    try:
        # Get credentials from Streamlit secrets
        creds_dict = get_google_credentials()
        if creds_dict is None:
            st.error("Cannot save to Google Drive: Missing credentials")
            return False
        
        # Set up credentials and authorize
        scopes = ['https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        # Your Google Drive file ID (same as in load_data)
        FILE_ID = "1IaVYJsgyqno73-O-s0TMD1AoQHRdkpsUz-LonB1CbF4"
        
        # Open the file and clear existing data
        file = gc.open_by_key(FILE_ID)
        worksheet = file.get_worksheet(0)
        worksheet.clear()
        
        # Prepare data for upload - convert all to strings to avoid formatting issues
        data_to_upload = [combined_df.columns.tolist()]  # Headers first
        data_to_upload.extend(combined_df.astype(str).values.tolist())  # Then data rows
        
        # Upload data to Google Sheets
        worksheet.update(data_to_upload)
        
        return True
        
    except Exception as e:
        st.error(f"Error saving to Google Drive: {str(e)}")
        return False

# Updated handle_file_upload function
def handle_file_upload():
    # Display different content based on upload stage
    if st.session_state.upload_success:
        st.success("✅ Dataset successfully updated!")
        if st.button("Upload Another File"):
            # Reset all states
            st.session_state.upload_stage = 'initial'
            st.session_state.processed_data = None
            st.session_state.upload_success = False
            st.experimental_rerun()
        return
        
    if st.session_state.upload_stage == 'initial':
        st.header("Upload New Data")
        
        # Data source selection using radio buttons
        upload_source = st.radio(
            "Select Organization Source",
            options=["LAET", "LAS", "WTLS", "MALS"],
            help="Select the organization this data is from"
        )
        
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
        
        if uploaded_file is not None:
            try:
                with st.spinner('Processing data...'):
                    # Read the uploaded file
                    df_new = pd.read_excel(uploaded_file, header=0)
                    
                    # === Header validation check ===
                    first_row_headers = df_new.columns.astype(str).str.strip()
                    known_headers = get_standard_mappings()[0].keys()

                    # Heuristic 1: Possible title row
                    has_title_row = (
                        (len(first_row_headers) <= 3 and any(first_row_headers.str.len() > 30)) or
                        (first_row_headers.isin(known_headers).sum() < 3)
                    )

                    # Heuristic 2: No recognizable column names
                    no_valid_columns = first_row_headers.isin(known_headers).sum() == 0

                    if has_title_row or no_valid_columns:
                        st.error(
                            "⚠️ It looks like the uploaded file either includes a header or is missing row with column names.\n\n"
                            "Please make sure:\n"
                            "- The first row contains column names (like 'Client ID', 'Date Opened')\n"
                            "- Any report titles or labels above the header row are removed"
                        )
                        return
                    # === End check ===

                    # Process data
                    df_processed = standardize_new_data(df_new, upload_source)
                    
                    # Store in session state
                    st.session_state.processed_data = df_processed
                    st.session_state.upload_stage = 'review'
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return
                
    elif st.session_state.upload_stage == 'review':
        st.header("Review Processed Data")
        
        df_processed = st.session_state.processed_data
        
        # Show preview in expandable section
        with st.expander("View Data Preview", expanded=True):
            st.write("First few rows of processed data:")
            st.dataframe(df_processed.head())
        
        try:
            # Load existing data from Google Drive
            with st.spinner('Loading existing dataset from Google Drive...'):
                existing_df = load_data()  # This now loads from Google Drive
            
            # Show metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("New Records", len(df_processed))
            with col2:
                st.metric("Total Records After Merge", len(existing_df) + len(df_processed))
            
            # Confirmation
            if st.button("Confirm and Save", type="primary"):
                with st.spinner('Saving data to Google Drive...'):
                    # Combine datasets
                    combined_df = pd.concat([existing_df, df_processed], ignore_index=True)
                    del existing_df, df_processed

                    # Make sure date columns are normalized before saving
                    date_columns = ['date_opened', 'date_closed']
                    for col in date_columns:
                        if col in combined_df.columns:
                            combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce').dt.normalize()

                    # Save combined data to Google Drive
                    if save_to_google_drive(combined_df):
                        # Clear the cache
                        del combined_df
                        st.cache_data.clear()
                        
                        # Update session state
                        st.session_state.upload_success = True
                        st.session_state.upload_stage = 'initial'
                        st.session_state.processed_data = None
                        st.experimental_rerun()
                    else:
                        st.error("Failed to save data to Google Drive. Please try again.")
            
            if st.button("Cancel"):
                st.session_state.upload_stage = 'initial'
                st.session_state.processed_data = None
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error processing existing data: {str(e)}")
            if st.button("Start Over"):
                st.session_state.upload_stage = 'initial'
                st.session_state.processed_data = None
                st.experimental_rerun()

def download_model_from_drive(file_id, model_name):
    """
    Download model from Google Drive using file ID
    """
    try:
        # Get credentials from Streamlit secrets
        creds_dict = get_google_credentials()
        if creds_dict is None:
            st.error(f"Cannot load {model_name} model: Missing credentials")
            return None
        
        # Set up credentials
        scopes = ['https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        
        # Download model using Google Drive API
        import googleapiclient.discovery
        service = googleapiclient.discovery.build('drive', 'v3', credentials=credentials)
        
        request = service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        
        import googleapiclient.http
        downloader = googleapiclient.http.MediaIoBaseDownload(file_content, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        file_content.seek(0)
        model = joblib.load(file_content)
        
        return model
        
    except Exception as e:
        st.error(f"Error loading {model_name} model from Google Drive: {str(e)}")
        return None

@st.cache_resource
def load_dv_model():
    """Load DV prediction model from Google Drive"""
    DV_MODEL_FILE_ID = "185zB1NckKot3BDv7uh-hUpgA8_-jTbuk"
    model = download_model_from_drive(DV_MODEL_FILE_ID, "DV prediction")
    return model

@st.cache_resource  
def load_case_time_model():
    """Load case time prediction model from Google Drive"""
    CASE_TIME_MODEL_FILE_ID = "12o4y0S9RiAFHmaoabASmYkQ2hWll03uZ"
    model = download_model_from_drive(CASE_TIME_MODEL_FILE_ID, "case time prediction")
    return model

# Load the data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load data with progress indicator
if not st.session_state.data_loaded:
    df = load_data()
    st.session_state.data_loaded = True
else:
    df = load_data()

# Add refresh button in sidebar
if st.sidebar.button('Refresh Data'):
    # Clear Streamlit's cache
    st.cache_data.clear()
    st.session_state.data_loaded = False
    st.experimental_rerun()

# Title and description
st.title("TALS Data Explorer")
st.write("Explore and analyze data across different legal service organizations")

# Sidebar filters
st.sidebar.header("Filters")

# Data source filter
def get_sorted_sources(df):
    if 'source' in df.columns:
        # Get unique sources and replace any unknown values
        sources = df['source'].dropna().unique() 
        return sorted(sources)
    return []

selected_sources = st.sidebar.multiselect(
    "Select Organization",
    options=get_sorted_sources(df),
    default=get_sorted_sources(df)
)

# Apply the filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['source'].isin(selected_sources)]

# Calculate date range based on actual data values - simply use normalized dates
date_opened_min = pd.to_datetime(df['date_opened']).min().date()
date_opened_max = pd.to_datetime(df['date_opened']).max().date()

# Create the date picker in sidebar
date_range = st.sidebar.date_input(
    "Select Date Opened Range",
    value=(date_opened_min, date_opened_max),
    min_value=date_opened_min,
    max_value=date_opened_max
)

# Calculate closed date range 
date_closed_min = pd.to_datetime(df['date_closed']).min().date()
date_closed_max = pd.to_datetime(df['date_closed']).max().date()

# Create the date picker in sidebar
closed_date_range = st.sidebar.date_input(
    "Select Date Closed Range",
    value=(date_closed_min, date_closed_max),
    min_value=date_closed_min,
    max_value=date_closed_max
)

# County filter section
st.sidebar.subheader("County Selection")

# Define Tennessee urban counties 
urban_counties = {
    'Davidson', 'Shelby', 'Knox', 'Hamilton', 'Rutherford', 
    'Williamson', 'Montgomery', 'Sumner', 'Wilson', 'Madison', 
    'Washington', 'Carter', 'Sullivan', 'Hawkins'
}

# Get all counties from the data
all_counties = sorted(df['county_dispute'].dropna().unique())

# Create separate lists for urban and rural counties
urban_county_list = sorted([county for county in all_counties if county in urban_counties])
rural_county_list = sorted([county for county in all_counties if county not in urban_counties])

# Create separate multiselect widgets for urban and rural counties
urban_counties_selected = st.sidebar.multiselect(
    "Select Urban Counties",
    options=urban_county_list,
    default=[]
)

rural_counties_selected = st.sidebar.multiselect(
    "Select Rural Counties",
    options=rural_county_list,
    default=[]
)

# Combine the selections
selected_counties = urban_counties_selected + rural_counties_selected

# Apply county filter if counties are selected
if selected_counties:
    filtered_df = filtered_df[filtered_df['county_dispute'].isin(selected_counties)]

# Handle date filtering
# Create a mask for valid dates (not null)
date_mask = filtered_df['date_opened'].notna()

# Split the dataframe into rows with valid dates and rows with unknown/null dates
date_df = filtered_df[date_mask].copy()
unknown_df = filtered_df[~date_mask].copy()

# Apply date filter only to rows with valid dates
if not date_df.empty:
    # Apply date range filter using the date_range from the sidebar
    date_filtered = date_df[
        (date_df['date_opened'].dt.date >= date_range[0]) &
        (date_df['date_opened'].dt.date <= date_range[1])
    ]
    
    # Combine filtered date rows with unknown date rows
    filtered_df = pd.concat([date_filtered, unknown_df])

# Create a mask for valid closed dates (not null)
closed_date_mask = filtered_df['date_closed'].notna()

# Split the dataframe into rows with valid closed dates and rows with unknown/null closed dates
closed_date_df = filtered_df[closed_date_mask].copy()
closed_unknown_df = filtered_df[~closed_date_mask].copy()

# Apply date filter only to rows with valid closed dates
if not closed_date_df.empty:
    # Apply date range filter using the closed_date_range from the sidebar
    closed_date_filtered = closed_date_df[
        (closed_date_df['date_closed'].dt.date >= closed_date_range[0]) &
        (closed_date_df['date_closed'].dt.date <= closed_date_range[1])
    ]
    
    # Update filtered_df to include the filtered closed date rows and unknown closed date rows
    filtered_df = pd.concat([closed_date_filtered, closed_unknown_df])

# Apply county filter if counties are selected
if selected_counties:
    filtered_df = filtered_df[filtered_df['county_dispute'].isin(selected_counties)]

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Overview", "Demographic Analysis", "Case Analysis", "Trends & Patterns", "Custom Visualization", "DV Risk Predictor", "Case Time Predictor", "Data Upload"])

with tab1:
    st.header("Overview Statistics")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cases", filtered_df['case_id'].nunique())
    with col2:
        st.metric("Average Days Open", round(filtered_df['days_open'].mean(), 1))
    with col3:
        st.metric("Unique Counties", filtered_df['county_dispute'].nunique())
    with col4:
        st.metric("Total Clients", filtered_df['client_id'].nunique())
    
    # Cases over time
    st.subheader("Cases Over Time")
    # Filter out unknown dates for the time series
    valid_dates_df = filtered_df[filtered_df['date_opened'].notna()].copy()
    if not valid_dates_df.empty:
        cases_over_time = valid_dates_df.groupby(
            pd.to_datetime(valid_dates_df['date_opened']).dt.to_period('M')
        ).size().reset_index(name='count')
        cases_over_time['date_opened'] = cases_over_time['date_opened'].astype(str)
        
        fig = px.line(cases_over_time, x='date_opened', y='count',
            title="Number of Cases Opened by Month",
            labels={'date_opened': 'Date Opened', 'count': 'Number of Cases'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No valid dates available for time series visualization")

with tab2:
    st.header("Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        st.subheader("Age Distribution")
        unique_age_dist = filtered_df.groupby('client_id')['age_intake'].first()
        fig = px.histogram(unique_age_dist, 
                  title="Age Distribution at Intake (Unique Clients)",
                  labels={'age_intake': 'Age at Intake', 'count': 'Number of Clients'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender distribution
        st.subheader("Gender Distribution")
        gender_counts = filtered_df.groupby('client_id')['gender'].first().value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index,
            title="Gender Distribution (Unique Clients)",
            labels={'names': 'Gender', 'values': 'Number of Clients'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Race distribution
        st.subheader("Race Distribution")
        race_counts = filtered_df.groupby('client_id')['race'].first().value_counts()
        fig = px.bar(x=race_counts.index, y=race_counts.values,
            title="Race Distribution (Unique Clients)",
            labels={'x': 'Race', 'y': 'Number of Clients'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Household size distribution
        st.subheader("Household Size Distribution")
        unique_household_dist = filtered_df.groupby('client_id')['household_total'].first()
        fig = px.histogram(unique_household_dist,
                  title="Household Size Distribution (Unique Clients)",
                  labels={'household_total': 'Household Total Size', 'count': 'Number of Clients'})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Case Analysis")
    
    # Global Food Stamps toggle for tab3
    exclude_foodstamps = st.checkbox("Exclude Food Stamps Cases (WTLS Counsel and Advice/Brief Service)", value=False, key="tab3_foodstamps_toggle")
    
    # Create filtered dataframe based on food stamps toggle
    display_df = filtered_df.copy()
    if exclude_foodstamps:
        display_df = display_df[~(
            (display_df['source'] == 'WTLS') & 
            (display_df['legal_problem_code'].str.contains('73 Food Stamps', case=False, na=False)) &
            (display_df['close_reason'].isin(['Counsel and Advice', 'X1-Brief Service']))
        )]

    
    # Legal problem codes
    st.subheader("Top Legal Problems")
    
    # Get problem counts
    problem_counts = display_df['legal_problem_code'].value_counts().head(10)
    
    fig = px.bar(x=problem_counts.index, y=problem_counts.values,
            title=f"Top 10 Legal Problems {'(Excluding Food Stamps)' if exclude_foodstamps else '(Including Food Stamps)'}",
            labels={'x': 'Legal Problem Code', 'y': 'Number of Cases'})
    st.plotly_chart(fig, use_container_width=True)

    # County heat map
    st.subheader("Geographic Distribution")

    # Volume selection radio button
    view_option = st.radio("Select View", [
        "High Volume (1000+ cases)",
        "Medium Volume (100-999 cases)",
        "Low Volume (<100 cases)"
    ], horizontal=True)

    # Calculate county counts from filtered data
    county_counts = display_df['county_dispute'].value_counts()

    # Filter counties based on volume selection
    if view_option == "High Volume (1000+ cases)":
        county_counts = county_counts[county_counts >= 1000]
        title = "Counties with 1000+ Cases"
    elif view_option == "Medium Volume (100-999 cases)":
        county_counts = county_counts[(county_counts >= 100) & (county_counts < 1000)]
        title = "Counties with 100-999 Cases"
    else:
        county_counts = county_counts[county_counts < 100]
        title = "Counties with Less Than 100 Cases"

    # Update title based on food stamps inclusion
    title += f" {'(Excluding Food Stamps)' if exclude_foodstamps else '(Including Food Stamps)'}"

    fig = go.Figure(data=go.Bar(x=county_counts.index, y=county_counts.values))
    fig.update_layout(title=title, xaxis_tickangle=-45,
                     xaxis_title="County of Dispute", yaxis_title="Number of Cases")
    st.plotly_chart(fig, use_container_width=True)

    # Case Duration Analysis
    st.subheader("Case Duration Analysis")

    # Create time-to-resolution analysis
    temporal_df = display_df.copy()
    temporal_df['date_opened'] = pd.to_datetime(temporal_df['date_opened'])
    temporal_df['date_closed'] = pd.to_datetime(temporal_df['date_closed'])
    temporal_df['resolution_time'] = (temporal_df['date_closed'] - temporal_df['date_opened']).dt.days

    # Remove any invalid resolution times (negative or zero)
    temporal_df = temporal_df[temporal_df['resolution_time'] > 0]

    # Check if we have enough data for duration analysis
    if len(temporal_df) < 4:  # Need at least 4 cases for quartile analysis
        st.warning(f"⚠️ Insufficient data for duration analysis. Only {len(temporal_df)} cases with valid resolution times found.")
        if len(temporal_df) > 0:
            st.info("**Basic Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cases", len(temporal_df))
            with col2:
                st.metric("Avg Days", f"{temporal_df['resolution_time'].mean():.1f}")
            with col3:
                st.metric("Max Days", f"{temporal_df['resolution_time'].max():.0f}")
        else:
            st.info("No cases with both open and close dates found. Try expanding your filters.")
        
    else:
        # Calculate quartiles and check for uniqueness
        duration_quartiles = temporal_df['resolution_time'].quantile([0.25, 0.5, 0.75])
        unique_values = temporal_df['resolution_time'].nunique()
        
        # Check if we can create meaningful bins
        can_create_quartiles = (
            unique_values >= 4 and  # At least 4 unique values
            len(set(duration_quartiles.values)) >= 3  # At least 3 unique quartile values
        )
        
        if not can_create_quartiles:
            # Fall back to simpler categorization
            st.info(f"📊 **Simple Duration Analysis** (Cases have limited duration variation)")
            
            # Show basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cases", len(temporal_df))
            with col2:
                st.metric("Avg Days", f"{temporal_df['resolution_time'].mean():.1f}")
            with col3:
                st.metric("Median Days", f"{temporal_df['resolution_time'].median():.1f}")
            with col4:
                st.metric("Range", f"{temporal_df['resolution_time'].min():.0f} - {temporal_df['resolution_time'].max():.0f}")
            
            # Create simple categorization based on overall mean
            overall_mean = temporal_df['resolution_time'].mean()
            temporal_df['duration_category'] = temporal_df['resolution_time'].apply(
                lambda x: 'Quick Resolution' if x <= overall_mean else 'Extended Resolution'
            )
            
            # Show distribution by simple categories
            category_dist = temporal_df['duration_category'].value_counts()
            st.write("**Case Distribution:**")
            for category, count in category_dist.items():
                st.write(f"- {category}: {count} cases ({count/len(temporal_df)*100:.1f}%)")
            
            # Show top legal problems in each category
            if len(temporal_df) >= 5:  # Only show breakdown if we have enough cases
                duration_type_dist = temporal_df.groupby(['duration_category', 'legal_problem_code']).size().reset_index(name='count')
                duration_type_dist['percentage'] = duration_type_dist.groupby('duration_category')['count'].transform(lambda x: (x/x.sum()) * 100)
                
                # Show top issues by simple duration category
                top_issues_simple = duration_type_dist.sort_values('count', ascending=False).head(10)
                
                fig = px.bar(
                    top_issues_simple,
                    x='duration_category',
                    y='count',
                    color='legal_problem_code',
                    title='Legal Issues by Duration Category (Simplified)',
                    labels={'count': 'Number of Cases', 'duration_category': 'Duration Category', 'legal_problem_code': 'Legal Problem Code'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Full quartile-based analysis
            st.success(f"📊 **Detailed Duration Analysis** ({len(temporal_df)} cases)")
            
            # Create robust bins that handle edge cases
            q25, q50, q75 = duration_quartiles[0.25], duration_quartiles[0.5], duration_quartiles[0.75]
            
            # Ensure bins are truly unique by adding small offsets if needed
            bins = [0]
            if q25 > 0:
                bins.append(q25)
            if q50 > q25:
                bins.append(q50)
            if q75 > q50:
                bins.append(q75)
            bins.append(float('inf'))
            
            # Create labels based on actual number of bins
            if len(bins) == 3:  # Only 2 categories
                labels = ['Quick Resolution', 'Extended Resolution']
            elif len(bins) == 4:  # 3 categories
                labels = ['Quick Resolution', 'Moderate Duration', 'Extended Resolution']
            else:  # 4 categories (full quartiles)
                labels = ['Quick Resolution', 'Moderate Duration', 'Extended Duration', 'Long-Term Cases']
            
            # Create duration categories
            temporal_df['duration_category'] = pd.cut(
                temporal_df['resolution_time'],
                bins=bins,
                labels=labels[:len(bins)-1],
                duplicates='drop'  # Handle any remaining duplicates
            )
            
            # Display quartile information
            st.write(f"""
            **Duration Ranges:**
            """)
            
            # Show actual ranges based on bins created
            for i, label in enumerate(labels[:len(bins)-1]):
                if i == 0:
                    range_desc = f"< {bins[i+1]:.0f} days"
                elif i == len(labels[:len(bins)-1]) - 1:
                    range_desc = f"> {bins[i]:.0f} days"
                else:
                    range_desc = f"{bins[i]:.0f} - {bins[i+1]:.0f} days"
                st.write(f"- {label}: {range_desc}")
            
            # Analyze what types of cases fall into each duration category
            duration_type_dist = temporal_df.groupby(['duration_category', 'legal_problem_code']).size().reset_index(name='count')
            duration_type_dist['percentage'] = duration_type_dist.groupby('duration_category')['count'].transform(lambda x: (x/x.sum()) * 100)
            
            # Show top issues in each duration category
            top_issues_by_duration = duration_type_dist.sort_values('percentage', ascending=False).groupby('duration_category').head(5)
            
            fig = px.bar(
                top_issues_by_duration,
                x='duration_category',
                y='percentage',
                color='legal_problem_code',
                title='Top Legal Issues by Case Duration Category',
                labels={'percentage': 'Percentage of Cases', 'duration_category': 'Duration Category', 'legal_problem_code': 'Legal Problem Code'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Global filter for Food Stamps
    exclude_foodstamps = st.checkbox("Exclude Food Stamps Cases (WTLS Counsel and Advice/Brief Service)", value=False)
    
    # Apply food stamps filter
    display_df = filtered_df.copy()
    if exclude_foodstamps:
        display_df = display_df[~(
            (display_df['source'] == 'WTLS') & 
            (display_df['legal_problem_code'].str.contains('73 Food Stamps', case=False, na=False)) &
            (display_df['close_reason'].isin(['Counsel and Advice', 'X1-Brief Service']))
        )]
        
    # Urban/Rural Analysis
    st.subheader("Urban/Rural Analysis")
    
    # Define Tennessee urban counties 
    urban_counties = {
        'Davidson', 'Shelby', 'Knox', 'Hamilton', 'Rutherford', 
        'Williamson', 'Montgomery', 'Sumner', 'Wilson', 'Madison', 
        'Washington', 'Carter', 'Sullivan','Hawkins'
    }
    
    # Create urban/rural classification 
    display_df['area_type'] = display_df['county_dispute'].apply(
        lambda x: 'Urban' if pd.notna(x) and str(x).strip() in urban_counties else 'Rural'
    )
    
    # Create problem distribution by area type
    area_problems = display_df.groupby(['area_type', 'legal_problem_code']).size().reset_index(name='count')
    area_problems['percentage'] = area_problems.groupby('area_type')['count'].transform(
    lambda x: (x / x.sum()) * 100)

    # Display percentage distribution
    area_distribution = display_df['area_type'].value_counts(normalize=True).mul(100).round(1)
    st.write(f"Urban: {area_distribution.get('Urban', 0)}% | Rural: {area_distribution.get('Rural', 0)}%")

    
    # Get top problems by percentage for each area
    top_problems_by_area = (area_problems.sort_values(['area_type', 'count'], ascending=[True, False])
                        .groupby('area_type')
                        .head(10))
    
    fig = px.bar(top_problems_by_area, 
            x='count', 
            y='legal_problem_code',
            color='area_type',
            barmode='group',
            title='Top 10 Legal Problems by Urban/Rural Areas',
            labels={'count': 'Number of Cases', 
               'legal_problem_code': 'Legal Problem Code',
               'area_type': 'Area Type'})
    fig.update_layout(
    showlegend=True,
    xaxis_title="Number of Cases",
    yaxis_title="Legal Problem",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Age Group Analysis
    st.subheader("Age Group Analysis")
    
    # Create age groups
    bins = [0, 25, 35, 50, 65, float('inf')]
    labels = ['18-25', '26-35', '36-50', '51-65', '65+']
    display_df['age_group'] = pd.cut(display_df['age_intake'], 
                                    bins=bins, 
                                    labels=labels,
                                    include_lowest=True)
    
    # Create a categorical type with specified order
    display_df['age_group'] = pd.Categorical(display_df['age_group'], 
                                        categories=['18-25', '26-35', '36-50', '51-65', '65+'],
                                        ordered=True)
    
    # Analyze problems by age group
    age_problems = display_df.groupby(['age_group', 'legal_problem_code']).size().reset_index(name='count')
    top_problems_by_age = age_problems.sort_values('count', ascending=False).groupby('age_group').head(5)
    
    fig = px.bar(top_problems_by_age,
            x='age_group',
            y='count',
            color='legal_problem_code',
            title='5 Most Common Legal Problems by Age Group',
            category_orders={'age_group': ['18-25', '26-35', '36-50', '51-65', '65+']},
            labels={'age_group': 'Age Group', 'count': 'Number of Cases', 'legal_problem_code': 'Legal Problem Code'})
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title="Legal Problem Code",
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analyze problems by gender
    gender_problems = display_df.groupby(['gender', 'legal_problem_code']).size().reset_index(name='count')
    top_problems_by_gender = gender_problems.sort_values('count', ascending=False).groupby('gender').head(5)

    # Create a subheader for the new section
    st.subheader("Legal Problems by Gender")

    # Create the gender breakdown plot
    fig_gender = px.bar(top_problems_by_gender,
                x='gender',
                y='count',
                color='legal_problem_code',
                title='Top 5 Legal Problems by Gender',
                labels={'gender': 'Gender', 'count': 'Number of Cases', 'legal_problem_code': 'Legal Problem Code'})

    fig_gender.update_layout(
        showlegend=True,
        legend=dict(
            title="Legal Problem Code",
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        )
    )

    # Display the plot
    st.plotly_chart(fig_gender, use_container_width=True)

    # Co-occurrence Analysis
    @st.cache_data
    def calculate_cooccurrence_matrix(df):
        """
        Calculate co-occurrence matrix using vectorized operations
        Returns both the matrix and a DataFrame of problem frequencies
        """
        # First get unique client-problem combinations
        unique_client_problems = df.drop_duplicates(['client_id', 'legal_problem_code'])
        
        # Create pivot table of client-problem relationships
        pivot = pd.crosstab(unique_client_problems['client_id'], 
                            unique_client_problems['legal_problem_code'])
        
        # Calculate co-occurrence matrix using matrix multiplication
        cooccurrence = pivot.T @ pivot
        
        # Get problem frequencies (number of unique clients with each problem)
        problem_frequencies = pivot.sum()
        
        return cooccurrence, problem_frequencies

    def display_cooccurrence_analysis(display_df):
        """Display co-occurrence analysis with optimized computations"""
        st.subheader("Legal Problem Co-occurrence Analysis")
        
        # Calculate co-occurrence matrix
        cooccurrence_matrix, problem_frequencies = calculate_cooccurrence_matrix(display_df)
        
        # Sort problems by frequency for better selectbox organization
        sorted_problems = problem_frequencies.index.tolist()
        
        # Create a selectbox with problems sorted by frequency
        selected_problem = st.selectbox(
            "Select a Legal Problem to see co-occurring issues:",
            options=sorted_problems,
            format_func=lambda x: f"{x} ({problem_frequencies[x]:,} cases)"
        )
        
        if selected_problem:
            # Get co-occurrences for selected problem
            cooccurrences = cooccurrence_matrix[selected_problem].sort_values(ascending=False)
            
            # Remove self-reference and zero counts
            cooccurrences = cooccurrences[
                (cooccurrences.index != selected_problem) & 
                (cooccurrences > 0)
            ]
            
            # Calculate percentages
            total_cases = problem_frequencies[selected_problem]
            cooccurrence_pcts = (cooccurrences / total_cases * 100).round(1)
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Co-occurring Problem': cooccurrences.index,
                'Number of Clients with Both Issues': cooccurrences.values,
                'Percentage of Clients with Both Issues': cooccurrence_pcts.values
            })
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Unique Clients With Selected Issue", f"{total_cases:,}")
            with col2:
                st.metric("Co-occurring Problems", f"{len(cooccurrences):,}")
            with col3:
                most_common = results_df.iloc[0]['Co-occurring Problem'] if not results_df.empty else "None"
                st.metric("Most Common Co-occurrence", most_common)
            
            # Display results
            st.markdown(f"""
            #### Co-occurrence Analysis for: {selected_problem}
            The table below shows other legal problems that appear for clients who have a {selected_problem} case.
            """)
            
            # Create styled dataframe
            styled_df = results_df.style\
                .format({
                    'Number of Clients with Both Issues': '{:,.0f}',
                    'Percentage of Clients with Both Issues': '{:.1f}%'
                })\
                .background_gradient(subset=['Percentage of Clients with Both Issues'], cmap='Blues')
            
            # Display the styled table
            st.dataframe(styled_df, width=800, height=400)
            
            # Add download option for the co-occurrence data
            st.download_button(
                "Download Co-occurrence Data",
                results_df.to_csv(index=False),
                file_name=f"cooccurrence_{selected_problem.replace('/', '_')}.csv",
                mime="text/csv"
            )
    display_cooccurrence_analysis(display_df)

with tab5:

    # Global Food Stamps toggle
    exclude_foodstamps = st.checkbox(
        "Exclude Food Stamps Cases (WTLS Counsel and Advice)", 
        value=False, 
        key="viz_tab_foodstamps"
    )

    display_df = filtered_df.copy()
    if exclude_foodstamps:
        display_df = display_df[~(
            (display_df['source'] == 'WTLS') & 
            (display_df['legal_problem_code'].str.contains('73 Food Stamps', case=False, na=False)) &
            (display_df['close_reason'].isin(['Counsel and Advice', 'X1-Brief Service']))
        )]

    st.subheader("Section A: Basic Plot Builder")

    basic_plot_type = st.selectbox("Select Basic Chart Type", [
        "Bar Chart",
        "Pie Chart", 
        "Line Chart",
        "Histogram",
        "Box Plot"
    ])

    safe_numeric_columns = [
        'age_intake', 
        'poverty_pct',
        'adj_poverty_pct',
        'household_total',
        'days_open',
        'case_time',
        'outcome_amount'
    ]

    safe_categorical_columns = [
        'county_dispute',
        'legal_problem_code',
        'race',
        'gender',
        'source',
        'domestic_violence',
        'income_eligible',
        'income_waiver_status',
        'asset_eligible'
    ]

    try:
        if basic_plot_type == "Bar Chart":
            category_col = st.selectbox("Select Category", options=safe_categorical_columns)
            value_counts = display_df[category_col].value_counts()
            if category_col != 'county_dispute':
                value_counts = value_counts.head(10)
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {category_col.replace('_', ' ').title()}",
                labels={'x': category_col.replace('_', ' ').title(), 'y': 'Count'}
            )
            fig.update_xaxes(tickangle=45)

        elif basic_plot_type == "Pie Chart":
            category_col = st.selectbox("Select Category", options=safe_categorical_columns)
            value_counts = display_df[category_col].value_counts().head(10)
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {category_col.replace('_', ' ').title()}",
                labels={'names': category_col.replace('_', ' ').title(), 'values': 'Count'}
            )

        elif basic_plot_type == "Line Chart":
            valid_dates_df = display_df[display_df['date_opened'].notna()].copy()
            valid_dates_df['month_year'] = pd.to_datetime(valid_dates_df['date_opened']).dt.to_period('M')
            cases_by_month = valid_dates_df.groupby('month_year').size().reset_index()
            cases_by_month['month_year'] = cases_by_month['month_year'].astype(str)
            fig = px.line(
                cases_by_month,
                x='month_year',
                y=0,
                title="Cases Over Time",
                labels={'month_year': 'Month/Year', '0': 'Number of Cases'}
            )
            fig.update_xaxes(tickangle=45)

        elif basic_plot_type == "Histogram":
            numeric_col = st.selectbox("Select Numeric Column", options=safe_numeric_columns)
            fig = px.histogram(
                display_df,
                x=numeric_col,
                title=f"Distribution of {numeric_col.replace('_', ' ').title()}",
                nbins=30,
                labels={numeric_col: numeric_col.replace('_', ' ').title(), 'count': 'Number of Cases'}
            )

        elif basic_plot_type == "Box Plot":
            numeric_col = st.selectbox("Select Numeric Value (Y-axis)", options=safe_numeric_columns)
            category_col = st.selectbox("Select Category (X-axis)", options=safe_categorical_columns)
            fig = px.box(
                display_df,
                x=category_col,
                y=numeric_col,
                title=f"{numeric_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}",
                labels={category_col: category_col.replace('_', ' ').title(), 
                       numeric_col: numeric_col.replace('_', ' ').title()}
            )
            fig.update_xaxes(tickangle=45)

        fig.update_layout(
            title_x=0.5,
            margin=dict(t=50, l=50, r=50, b=50),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while creating the basic visualization: {str(e)}")

    # === Section B: Advanced Analysis ===
    st.markdown("---")
    st.subheader("Section B: Advanced Analysis")

    plot_focus = st.radio("What would you like to analyze?", 
                        ["Outcome Amount", "Legal Problems"])

    if plot_focus == "Outcome Amount":
        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Box Plot", "Histogram"])

        # Let users optionally remove zero-outcome cases
        exclude_zeros = st.checkbox("Exclude Outcome Amount = 0", value=True)

        df_plot = display_df.copy()
        df_plot["outcome_amount"] = pd.to_numeric(df_plot["outcome_amount"], errors="coerce")
        df_plot = df_plot[df_plot["outcome_amount"].notna()]
        if exclude_zeros:
            df_plot = df_plot[df_plot["outcome_amount"] > 0]

        if chart_type == "Histogram":
            st.info("Showing raw distribution of outcome amounts")
            fig = px.histogram(df_plot, x="outcome_amount", nbins=30,
                title="Distribution of Outcome Amount",
                labels={'outcome_amount': 'Outcome Amount', 'count': 'Number of Cases'})
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot":
            group_col = st.selectbox("Group by:", [
                "county_dispute", "gender", "race", "source",
                "income_eligible", "income_waiver_status", "asset_eligible"
            ])
            fig = px.box(df_plot, x=group_col, y="outcome_amount",
                        title=f"Outcome Amount by {group_col.replace('_', ' ').title()}",
                        labels={group_col: group_col.replace('_', ' ').title(), 'outcome_amount': 'Outcome Amount'})
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart":
            group_col = st.selectbox("Group by:", [
                "county_dispute", "gender", "race", "source",
                "income_eligible", "income_waiver_status", "asset_eligible"
            ])
            agg_func = st.selectbox("Aggregation Function", ["Mean", "Median", "Sum"])
            agg_map = {"Mean": "mean", "Median": "median", "Sum": "sum"}
            summary = df_plot.groupby(group_col)["outcome_amount"].agg(agg_map[agg_func]).reset_index()
            summary = summary[summary["outcome_amount"].notna() & (summary["outcome_amount"] > 0)]

            fig = px.bar(summary, x=group_col, y="outcome_amount",
                        title=f"{agg_func} Outcome Amount by {group_col.replace('_', ' ').title()}",
                        labels={group_col: group_col.replace('_', ' ').title(), "outcome_amount": f"{agg_func} Outcome Amount"})
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # === LEGAL PROBLEMS ANALYSIS ===
    elif plot_focus == "Legal Problems":
        analysis_type = st.selectbox("Choose analysis type", [
            "Distribution by Category", 
            "Numeric Analysis"
        ])

        # Let user choose specific legal problems
        codes = display_df['legal_problem_code'].dropna().unique()
        selected_codes = st.multiselect("Select Legal Problem Codes", sorted(codes), default=sorted(codes)[:10])
        display_df = display_df[display_df['legal_problem_code'].isin(selected_codes)]

        if analysis_type == "Distribution by Category":
            category = st.selectbox("Group by:", [
                "gender", "race", "county_dispute", "source",
                "income_eligible", "income_waiver_status", "asset_eligible"
            ])

            try:
                count_df = display_df.groupby(["legal_problem_code", category]).size().reset_index(name='count')
                fig = px.bar(
                    count_df,
                    x=category,
                    y="count",
                    color="legal_problem_code",
                    barmode='group',
                    title=f"Legal Problems by {category.replace('_', ' ').title()}",
                    labels={"count": "Number of Cases", category: category.replace('_', ' ').title(), "legal_problem_code": "Legal Problem Code"},
                    category_orders={"legal_problem_code": sorted(selected_codes)}
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")

        elif analysis_type == "Numeric Analysis":
            numeric_col = st.selectbox("Numeric Column", [
                "outcome_amount", "case_time", "age_intake", "poverty_pct", "adj_poverty_pct"
            ])

            df_plot = display_df.copy()
            df_plot[numeric_col] = pd.to_numeric(df_plot[numeric_col], errors="coerce")
            df_plot = df_plot[df_plot[numeric_col].notna()]

            fig = px.box(df_plot, x="legal_problem_code", y=numeric_col,
                        title=f"{numeric_col.replace('_', ' ').title()} by Legal Problem Code",
                        labels={numeric_col: numeric_col.replace('_', ' ').title(), "legal_problem_code": "Legal Problem Code"})
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("DV Risk Prediction Tool")
    st.write("Estimate the likelihood that a case involves domestic violence based on intake data. This tool is for guidance only and meant to help identify cases that may need additional screening.")

    # Import preprocessing functions
    from preprocessing import preprocess_client_data, interpret_risk_score

    # Load the model (use st.cache_resource to load it only once)
    try:
        model = load_dv_model()  # This now uses the Google Drive function
        model_loaded = True if model is not None else False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_loaded = False

    # Get unique values from dataset for dropdown options
    @st.cache_data
    def get_unique_options(df, column):
        if column in df.columns:
            options = df[column].dropna().unique()
            return sorted(options)
        return []

    with st.form("dv_form"):
        st.subheader("Client & Case Info")
        st.markdown("*All fields are required for accurate prediction*")

        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demographics**")
            # Demographic information
            age = st.number_input("Age at Intake", min_value=18, max_value=120, value=35, key="dv_age")
            gender = st.selectbox("Gender", get_unique_options(df, 'gender'), key="dv_gender")
            race = st.selectbox("Race", get_unique_options(df, 'race'), key="dv_race")
            disabled = st.selectbox("Disabled", get_unique_options(df, 'disabled'), key="dv_disabled")
            veteran = st.selectbox("Veteran", get_unique_options(df, 'veteran'), key="dv_veteran")
            
        with col2:
            st.markdown("**Household Information**")
            # Household information
            household_total = st.number_input("Total Household Size", min_value=1, max_value=15, value=2, key="dv_household")
            household_adults = st.number_input("Adults", min_value=1, max_value=10, value=1, key="dv_adults")
            household_children = st.number_input("Children", min_value=0, max_value=10, value=1, key="dv_children")
            living_arrangement = st.selectbox("Living Arrangement", 
                                            get_unique_options(df, 'living_arrangement'), key="dv_living")

        # Economic and location information
        st.markdown("**Economic & Location Information**")
        col3, col4 = st.columns(2)
        with col3:
            poverty_pct = st.number_input("Poverty % (Federal Poverty Level)", min_value=0.0, max_value=1000.0, value=100.0,
                                        help="100% = at poverty line, <100% = below poverty line", key="dv_poverty")
            adj_poverty_pct = st.number_input("Adjusted Poverty %", min_value=-500.0, max_value=500.0, value=100.0, key="dv_adj_poverty")
            zip_code = st.number_input("ZIP Code", min_value=10000, max_value=99999, value=37000, key="dv_zip")
        
        with col4:
            county_residence = st.selectbox("County of Residence", 
                                          get_unique_options(df, 'county_residence'), key="dv_county_res")
            county_dispute = st.selectbox("County of Dispute", 
                                        get_unique_options(df, 'county_dispute'), key="dv_county_disp")
            source = st.selectbox("Referral Source", get_unique_options(df, 'source'), key="dv_source")
        
        # Additional fields
        st.markdown("**Additional Information**")
        col5, col6 = st.columns(2)
        with col5:
            citizenship = st.selectbox("Citizenship", get_unique_options(df, 'citizenship'), key="dv_citizenship")
        with col6:
            language = st.selectbox("Language", get_unique_options(df, 'language'), key="dv_language")
        
        # Add validation
        validation_error = None
        if household_adults + household_children != household_total:
            validation_error = "⚠️ Total household size must equal adults + children"
        
        if validation_error:
            st.error(validation_error)
        
        submitted = st.form_submit_button("Predict Risk", disabled=(validation_error is not None))

    if submitted:
        if not model_loaded:
            st.error("Cannot generate prediction: Model not loaded")
        else:
            # Prepare client data dictionary with only the features used in the model
            client_data = {
                'age_intake': age,
                'household_total': household_total,
                'household_adults': household_adults,
                'household_children': household_children,
                'poverty_pct': poverty_pct,
                'adj_poverty_pct': adj_poverty_pct,
                'zip_code': zip_code,
                'gender': gender,
                'race': race,
                'disabled': disabled,
                'veteran': veteran,
                'county_residence': county_residence,
                'county_dispute': county_dispute,
                'living_arrangement': living_arrangement,
                'source': source,
                'citizenship': citizenship,
                'language': language
            }
            
            # Use preprocessing function from imported module
            processed_data = preprocess_client_data(client_data)
            
            # Try to make prediction
            try:
                risk_score = model.predict_proba(processed_data)[0, 1]
                
                # Use interpret_risk_score function from preprocessing module
                result = interpret_risk_score(risk_score)
                
                # Display results
                st.success("✅ DV Risk Assessment Complete")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Risk Assessment")
                    st.metric("Risk Score", f"{result['risk_score']:.3f}")
                    
                    # Color-coded risk level
                    if result['risk_level'] == "Low":
                        st.success(f"🟢 **Risk Level:** {result['risk_level']}")
                    elif result['risk_level'] == "Medium":
                        st.warning(f"🟡 **Risk Level:** {result['risk_level']}")
                    else:
                        st.error(f"🔴 **Risk Level:** {result['risk_level']}")
                        
                    st.markdown(f"**Recommendation:** {result['recommendation']}")
                    
                    # Show household composition indicator
                    single_parent = (household_adults == 1 and household_children > 0)
                    if single_parent:
                        st.info("👥 **Household Type:** Single Parent (risk factor)")
                    else:
                        st.info(f"👥 **Household Type:** {household_adults} adult(s), {household_children} child(ren)")
                
                with col2:
                    # Create a gauge visualization
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Score"},
                        gauge = {
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.4], 'color': "lightgreen"},
                                {'range': [0.4, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1.0], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_score
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model transparency section
                with st.expander("🔍 How this prediction was made"):
                    st.markdown(f"""
                    **Key factors in this assessment:**
                    - **Household Structure:** {'Single parent household' if single_parent else f'{household_adults} adults with {household_children} children'} 
                    - **Demographics:** Age ({age}), gender, race, veteran/disability status
                    - **Economic Factors:** Poverty level ({poverty_pct:.0f}% of federal poverty line)
                    - **Location:** ZIP code ({zip_code}), county information
                    - **Case Context:** Referral source and living arrangement
                    
                    **Model Performance:**
                    - ROC AUC Score: 0.79 (good discriminative ability)
                    - Recall: 75.4% (catches 3 out of 4 actual DV cases)
                    - Precision: 36.5% (about 1 in 3 high-risk predictions are actual DV cases)
                    - **Optimized for high recall** to minimize missing actual DV cases
                    - Based on {8500:,} historical cases
                    
                    **Risk Score Interpretation:**
                    - **0.0-0.4:** Low risk (similar to cases that typically don't involve DV)
                    - **0.4-0.7:** Medium risk (mixed indicators, additional screening recommended)
                    - **0.7-1.0:** High risk (similar to cases that historically involved DV)
                    """)
                
                # Risk level specific guidance
                st.subheader("General Screening Considerations")
                
                if result['risk_level'] == "Low":
                    st.success("📋 **Low Risk Level**")
                    st.markdown("""
                    Based on historical patterns, cases with similar characteristics typically do not involve domestic violence.
                    Standard intake procedures are appropriate.
                    """)
                elif result['risk_level'] == "Medium":
                    st.warning("🔍 **Medium Risk Level**")
                    st.markdown("""
                    Mixed risk indicators present. Historical cases with similar patterns show variable outcomes.
                    Additional screening questions during intake may be helpful.
                    """)
                else:
                    st.error("⚠️ **High Risk Level**")
                    st.markdown("""
                    Risk factors present that are similar to cases that historically involved domestic violence.
                    Consider additional screening and connecting to appropriate resources per organizational protocols.
                    """)
                
                # Important disclaimer
                st.info("""
                **Disclaimer:** This is a screening tool based on statistical patterns in historical data. 
                Domestic violence can occur across all demographics and circumstances. This tool should supplement, 
                not replace, professional judgment, direct client assessment, and established organizational protocols. 
                The model prioritizes identifying potential DV cases, which may result in false positives.
                """)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                with st.expander("Troubleshooting"):
                    st.markdown("""
                    **Common issues:**
                    - Make sure all dropdown fields have valid selections
                    - Verify household numbers add up correctly  
                    - Check that ZIP code is 5 digits
                    - Ensure age is within valid range (18-120)
                    - Verify poverty percentages are reasonable
                    """)

with tab7:  # Case Time Prediction tab
    st.header("Case Time Prediction")
    st.write("Estimate the expected duration of a case to help with planning.")

    # Import the prediction function from preprocessing.py
    from preprocessing import predict_case_time

    with st.form("case_time_form"):
        st.subheader("Client & Case Info")
        st.markdown("*All fields are required for accurate prediction*")

        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demographics**")
            # Core features used in model training
            age = st.number_input("Age at Intake", min_value=18, max_value=100, value=35, key="ct_age")
            gender = st.selectbox("Gender", get_unique_options(df, 'gender'), key="ct_gender")
            race = st.selectbox("Race", get_unique_options(df, 'race'), key="ct_race")
            disabled = st.selectbox("Disabled", get_unique_options(df, 'disabled'), key="ct_disabled")
            veteran = st.selectbox("Veteran", get_unique_options(df, 'veteran'), key="ct_veteran")
        
        with col2:
            st.markdown("**Household Information**")
            household_total = st.number_input("Total Household Size", min_value=1, max_value=15, value=2, key="ct_household")
            household_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=1, key="ct_adults") 
            household_children = st.number_input("Number of Children", min_value=0, max_value=10, value=1, key="ct_children")
            living_arrangement = st.selectbox("Living Arrangement", 
                                           get_unique_options(df, 'living_arrangement'), key="ct_living")
        
        # Economic information
        st.markdown("**Economic Information**")
        col3, col4 = st.columns(2)
        with col3:
            poverty_pct = st.number_input("Poverty % (Federal Poverty Level)", 
                                        min_value=0.0, max_value=1000.0, value=100.0, 
                                        help="100% = at poverty line, <100% = below poverty line",
                                        key="ct_poverty")
        with col4:
            adj_poverty_pct = st.number_input("Adjusted Poverty %", 
                                            min_value=-500.0, max_value=500.0, value=100.0,
                                            help="Adjusted poverty calculation",
                                            key="ct_adj_poverty")
        
        # Case and location information
        st.markdown("**Case Information**")
        col5, col6 = st.columns(2)
        with col5:
            county_residence = st.selectbox("County of Residence", 
                                         get_unique_options(df, 'county_residence'), key="ct_county_res")
            county_dispute = st.selectbox("County of Dispute", 
                                       get_unique_options(df, 'county_dispute'), key="ct_county_disp")
        with col6:
            source = st.selectbox("Referral Source", get_unique_options(df, 'source'), key="ct_source")
            # This is the most important feature for case time prediction
            legal_problem_code = st.selectbox("Legal Problem Code", 
                                            get_unique_options(df, 'legal_problem_code'), 
                                            help="Primary legal issue - this significantly affects case duration",
                                            key="ct_legal_code")
        
        # Add validation
        validation_error = None
        if household_adults + household_children != household_total:
            validation_error = "⚠️ Total household size must equal adults + children"
        
        if validation_error:
            st.error(validation_error)
        
        submitted = st.form_submit_button("Predict Case Time", disabled=(validation_error is not None))

    if submitted:
        # Prepare client data dictionary with exactly the features used in training
        client_data = {
            'age_intake': age,
            'household_total': household_total,
            'household_adults': household_adults,
            'household_children': household_children,
            'poverty_pct': poverty_pct,
            'adj_poverty_pct': adj_poverty_pct,
            'gender': gender,
            'race': race,
            'disabled': disabled,
            'veteran': veteran,
            'county_residence': county_residence,
            'county_dispute': county_dispute,
            'living_arrangement': living_arrangement,
            'source': source,
            'legal_problem_code': legal_problem_code
        }
        
        # Get prediction
        result = predict_case_time(client_data)
        
        if result['predicted_hours'] is not None:
            # Display results
            st.success("✅ Case Time Prediction Complete")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Time Estimate")
                st.metric("Predicted Case Hours", f"{result['predicted_hours']}")
                st.markdown(f"**Complexity:** {result['complexity_category']}")
                st.markdown(f"**Resource Consideration:** {result['resource_allocation']}")
                
                # Show which legal problem category this falls into
                # Determine legal problem group (same logic as in preprocessing)
                code_str = str(legal_problem_code).strip()
                if code_str.startswith('0') or code_str.startswith('1'):
                    legal_group = 'Consumer/Finance'
                elif code_str.startswith('12') or code_str.startswith('13') or code_str.startswith('14') or code_str.startswith('16') or code_str.startswith('19'):
                    legal_group = 'Education'
                elif code_str.startswith('2'):
                    legal_group = 'Employment'
                elif code_str.startswith('3'):
                    legal_group = 'Family Law'
                elif code_str.startswith('4'):
                    legal_group = 'Juvenile'
                elif code_str.startswith('5'):
                    legal_group = 'Health'
                elif code_str.startswith('6'):
                    legal_group = 'Housing'
                elif code_str.startswith('7'):
                    legal_group = 'Income/Benefits'
                elif code_str.startswith('8'):
                    legal_group = 'Civil Rights'
                elif code_str.startswith('9'):
                    legal_group = 'Miscellaneous'
                else:
                    legal_group = 'Other'
                
                st.info(f"📋 **Legal Category:** {legal_group}")
            
            with col2:
                # Create a gauge visualization
                import plotly.graph_objects as go
                
                # Set reasonable limits for gauge
                max_display = 40  # Cap the gauge display
                predicted_hours = result['predicted_hours']
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge",
                    value = min(predicted_hours, max_display),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Estimated Hours"},
                    gauge = {
                        'axis': {'range': [0, max_display]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 3], 'color': "lightgreen"},
                            {'range': [3, 10], 'color': "yellow"},
                            {'range': [10, max_display], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': predicted_hours
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                
                if predicted_hours > max_display:
                    st.caption(f"Note: Actual prediction is {predicted_hours:.1f} hours (gauge capped at {max_display})")
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Resource planning with specific recommendations
            st.subheader("Planning Considerations")
            
            if result['complexity_category'] == "Brief Service":
                st.success("📋 **Brief Service Case** (Under 3 hours)")
    
            elif result['complexity_category'] == "Moderate Complexity":
                st.warning("⚖️ **Moderate Complexity Case** (3-10 hours)")

            else:
                st.error("📊 **High Complexity Case** (10+ hours)")
            
            # Model transparency
            with st.expander("🔍 How this prediction was made"):
                st.markdown(f"""
                **Key factors in this prediction:**
                - **Legal Problem Type:** {legal_group} cases tend to have specific duration patterns
                - **Client Demographics:** Age ({age}), household size ({household_total}), poverty level
                - **Case Location:** County match and referral source
                - **Household Complexity:** Adult-to-child ratio and household structure
                
                **Model Performance:**
                - R² Score: 0.78 (explains 78% of case duration variance)
                - Typical prediction accuracy: ±5.8 hours
                - Based on {37201:,} historical cases
                """)
            
            # Add disclaimer
            st.info("""
            **Important Note:** This prediction is based on historical data patterns and should be used 
            for planning purposes only. Actual case times may vary significantly based on various factors
            """)
        else:
            st.error("❌ Error making prediction. Please check that all fields are filled correctly and try again.")
            with st.expander("Troubleshooting"):
                st.markdown("""
                **Common issues:**
                - Make sure all dropdown fields have valid selections
                - Verify household numbers add up correctly
                - Check that poverty percentages are reasonable
                - Ensure age is within valid range (18-100)
                """)

with tab8:
    handle_file_upload()

# Add download buttons for filtered data
st.sidebar.markdown("---")
st.sidebar.header("Download Filtered Data")

# Download CSV button
st.sidebar.download_button(
    label="Download as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# Download Excel button
if st.sidebar.button("Prepare Excel Download"):
    with st.sidebar.spinner('Preparing Excel file...'):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False)
        buffer.seek(0)
        
        st.sidebar.download_button(
            label="Download Excel File",
            data=buffer,
            file_name="filtered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
