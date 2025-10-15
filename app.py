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
from preprocessing import preprocess_client_data, interpret_risk_score, predict_case_time_with_model, predict_case_time
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
            st.session_state["current_username"] = username
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
        'Caucasian': 'White',
        'White - Not Hispanic': 'White',
        'white': 'White',
        
        # Black categories
        'Black': 'Black',
        'Black (Not Hispanic)': 'Black',
        'African American/Black': 'Black',
        'Black or African American': 'Black',
        'Black - Not Hispanic': 'Black',
        'African-American': 'Black',
        'African American': 'Black',
        'AA': 'Black',
        'black': 'Black',
        
        # Native American categories
        'Native American': 'Native American',
        'Native American or Alaska Native': 'Native American',
        'American Indian or Alaska Native': 'Native American',
        'American Indian or Alaska Native and White': 'Native American',
        'American Indian or Alaska Native anc': 'Native American',  # Your truncated version
        
        # Asian and Pacific Islander categories
        'Asian': 'Asian/Pacific Islander',
        'Asian or Pacific Islander': 'Asian/Pacific Islander',
        'Asian/Pacific Islander': 'Asian/Pacific Islander',
        'Native Hawaiian or Other Pacific Islander': 'Asian/Pacific Islander',
        
        # Hispanic
        'Hispanic': 'Hispanic',
        'hispanic': 'Hispanic',
        
        # Multiracial categories
        'Multiracial': 'Multiracial',
        'Mulitracial': 'Multiracial',
        'Multi-Racial': 'Multiracial',
        'Black or African American and White': 'Multiracial',
        'Asian and White': 'Multiracial',
        
        # Other categories
        'Other': 'Other/Unknown',
        'Other/Unknown': 'Other/Unknown',
        'Other Ethnic Group': 'Other/Unknown',
        'No Response': 'Other/Unknown',
        'Organization/Group': 'Other/Unknown',
        '': 'Other/Unknown',
        'nan': 'Other/Unknown',
        None: 'Other/Unknown'
    }
    
    # Gender mapping
    gender_mapping = {
        # Female
        'Female': 'Female',
        'female': 'Female',
        'F': 'Female',
        'Woman': 'Female',
        
        # Male
        'Male': 'Male',
        'male': 'Male',
        'M': 'Male',
        'Man': 'Male',
        
        # Transgender
        'Transgender Female to Male': 'Transgender',
        'Transgender Male to Female': 'Transgender',
        'Trans man': 'Transgender',
        'Trans woman': 'Transgender',
        'Transgender': 'Transgender',
        'Trans': 'Transgender',
        
        # Non-binary
        'Non-binary': 'Non-binary',
        'Non-Binary': 'Non-binary',
        'Nonbinary': 'Non-binary',
        'Gender Non-Conforming': 'Non-binary',
        'Genderqueer': 'Non-binary',
        
        # Other/Unknown
        "Don't Know": 'Other/Unknown',
        'Other': 'Other/Unknown',
        'Prefer not to say': 'Other/Unknown',
        'Decline to state': 'Other/Unknown',
        'G': 'Other/Unknown',
        '7': 'Other/Unknown',
        'nan': 'Other/Unknown',
        '': 'Other/Unknown',
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
    r'(?i)^\s*1?5\s*[-: ]?\s*.*?(?:vocat.*?ed)': '15 Vocational Education',
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
    return column_mapping, race_mapping, gender_mapping, legal_problem_mapping

# Function to apply regex-based legal problem mapping
def map_legal_problem_with_regex(problem_code, legal_problem_patterns):
    """
    Map legal problem codes to standardized format using multi-tiered approach:
    1. Direct mapping (fastest)
    2. Regex patterns (flexible)
    3. Numeric code fallback (catches unknown variations)
    """
    if pd.isna(problem_code):
        return None
    
    # Convert to string and strip whitespace
    problem_str = str(problem_code).strip()
    
    # Extract numeric code at start (e.g., "05", "62") for fallback matching
    code_match = re.match(r'^\s*0*(\d+)', problem_str)
    numeric_code = code_match.group(1).zfill(2) if code_match else None
    
    # Normalize for case-insensitive matching
    normalized = problem_str.lower()
    
    # Direct standardization mapping - MOST EFFICIENT, TRIES FIRST
    standardization_map = {
        # Consumer/Finance (01-09)
        '01 bankruptcy/debtor relief': '01 Bankruptcy/Debtor Relief',
        '02 collection (including repo/def/garnish)': '02 Collection (including Repo/Def/Garnish)',
        '02 collect/repo/def/garnsh': '02 Collection (including Repo/Def/Garnish)',
        '02 - collections (repo, def., garn)': '02 Collection (including Repo/Def/Garnish)',
        '03 contracts / warranties': '03 Contracts/Warranties',
        '03 contracts/warranties': '03 Contracts/Warranties',
        '03 contract/warranties': '03 Contracts/Warranties',
        '04 collection practices/creditor harassment': '04 Collection Practices/Creditor Harassment',
        '04 collection practices / creditor harassment': '04 Collection Practices/Creditor Harassment',
        '05 predatory lending practices (not mortgages)': '05 Predatory Lending Practices (not mortgages)',
        '06 loans/installment purch.': '06 Loans/Installment Purch.',
        '06 loans/installment purchases (not collections)': '06 Loans/Installment Purch.',
        '07 public utilities': '07 Public Utilities',
        '08 unfair and deceptive sales and practices (not real property)': '08 Unfair and Deceptive Sales and Practices (not real property)',
        '08 unfair and deceptive sales practices (not real property)': '08 Unfair and Deceptive Sales and Practices (not real property)',
        '09 other consumer/finance': '09 Other Consumer/Finance',
        '09 other consumer / finance.': '09 Other Consumer/Finance',

        # Education (12-19)
        '12 discipline (including expulsion and suspension)': '12 Discipline (including expulsion and suspension)',
        '13 special education/learning disabilities': '13 Special Education/Learning Disabilities',
        '14 access (including bilingual, residency, testing)': '14 Access (Including Bilingual, Residency, Testing)',
        '15 vocational education': '15 Vocational Education',
        '16 student financial aid': '16 Student Financial Aid',
        '19 other education': '19 Other Education',

        # Employment (21-29)
        '21 employment discrimination': '21 Employment Discrimination',
        '22 wage claim and other flsa issues': '22 Wage Claim and other FLSA Issues',
        '22 wage claims and other flsa issues': '22 Wage Claim and other FLSA Issues',
        '23 eitc (earned income tax credit)': '23 EITC (Earned Income Tax Credit)',
        '24 taxes (not eitc)': '24 Taxes (not EITC)',
        '25 employee rights': '25 Employee Rights',
        '29 other employment & ceta': '29 Other Employment',
        '29 other employment': '29 Other Employment',

        # Family (30-39)
        '30 adoption': '30 Adoption',
        '31 custody/visitation': '31 Custody/Visitation',
        '31 custody / visitation': '31 Custody/Visitation',
        '32 divorce/sep./annul.': '32 Divorce/Sep./Annul.',
        '32 divorce / sep. / annul.': '32 Divorce/Sep./Annul.',
        '33 adult guardianship / conserv.': '33 Adult Guardianship/Conserv.',
        '33 adult guardianship/conserv.': '33 Adult Guardianship/Conserv.',
        '33 adult guardianship / conservatorship': '33 Adult Guardianship/Conserv.',
        '34 name change': '34 Name Change',
        '35 parental rights termin.': '35 Parental Rights Termin.',
        '35 parental rights termination': '35 Parental Rights Termin.',
        '36 paternity': '36 Paternity',
        '37 domestic abuse': '37 Domestic Abuse',
        '37 - domestic abuse': '37 Domestic Abuse',
        '38 support': '38 Support',
        '39 other family': '39 Other Family',

        # Juvenile (41-49)
        '41 delinquent': '41 Delinquent',
        '42 neglected/abused/depend.': '42 Neglected/Abused/Depend.',
        '42 neglected/abused/dependent': '42 Neglected/Abused/Depend.',
        '43 emancipation': '43 Emancipation',
        '44 minor guardian/conservatorship': '44 Minor Guardian/Conservatorship',
        '44 minor guardianship / conservatorship': '44 Minor Guardian/Conservatorship',
        '49 other juvenile': '49 Other Juvenile',

        # Health (51-59)
        '51 medicaid': '51 Medicaid',
        '51 - medicaid (tenncare)': '51 Medicaid',
        '52 medicare': '52 Medicare',
        "53 government children's health insurance programs": "53 Government Children's Health Insurance Programs",
        "53 goverment children's health insurance programs": "53 Government Children's Health Insurance Programs",
        '54 home and community based care': '54 Home and Community Based Care',
        '55 private health insurance': '55 Private Health Insurance',
        '56 long term health care facilities': '56 Long Term Health Care Facilities',
        '57 state and local health': '57 State and Local Health',
        '59 other health': '59 Other Health',

        # Housing (61-69)
        '61 fed. subsidized housing': '61 Fed. Subsidized Housing',
        '61 federally subsidized housing': '61 Fed. Subsidized Housing',
        '61 - federally subsidized housing': '61 Fed. Subsidized Housing',
        '62 homeownership/real prop. (not foreclosure)': '62 Homeownership/Real Prop. (not foreclosure)',
        '62 homeownership/real property (not foreclosure)': '62 Homeownership/Real Prop. (not foreclosure)',
        '63 private landlord / tenant': '63 Private Landlord/Tenant',
        '63 private landlord/tenant': '63 Private Landlord/Tenant',
        '63 - private landlord/tenant': '63 Private Landlord/Tenant',
        '64 public housing': '64 Public Housing',
        '65 mobile homes': '65 Mobile Homes',
        '66 housing discrimination': '66 Housing Discrimination',
        '67 mortgage foreclosures (not predatory lending/practices)': '67 Mortgage Foreclosures (not predatory Lending/practices)',
        '68 mortgage predatory lending/practices': '68 Mortgage Predatory Lending/Practices',
        '69 other housing': '69 Other Housing',

        # Income Maintenance (71-79)
        '71 tanf': '71 TANF',
        '71 - tanf (families first)': '71 TANF',
        '72 social security (not ssdi)': '72 Social Security (not SSDI)',
        '73 food stamps': '73 Food Stamps',
        '73 food stamps / commodities': '73 Food Stamps',
        '74 ssdi': '74 SSDI',
        '75 ssi': '75 SSI',
        '76 unemployment compensation': '76 Unemployment Compensation',
        '77 veterans benefits': '77 Veterans Benefits',
        '78 state and local income maintenance': '78 State and Local Income Maintenance',
        '79 other income maintenance': '79 Other Income Maintenance',
        '79 other income maintenence': '79 Other Income Maintenance',

        # Rights and Other (81-89)
        '81 immigration/naturalization': '81 Immigration/Naturalization',
        '81 immigration / naturalization': '81 Immigration/Naturalization',
        '82 mental health': '82 Mental Health',
        '84 disability rights': '84 Disability Rights',
        '85 civil rights': '85 Civil Rights',
        '86 human trafficking': '86 Human Trafficking',
        '87 expungement': '87 Expungement',
        '87 - expungement': '87 Expungement',
        '87 criminal record expungement': '87 Expungement',
        '89 other individual rights': '89 Other Individual Rights',

        # Miscellaneous (93-99)
        '93 licenses (auto and other)': '93 Licenses (Auto and Other)',
        '93 licenses (drivers, occupational, and others)': '93 Licenses (Auto and Other)',
        '94 torts': '94 Torts',
        '95 wills / estates': '95 Wills/Estates',
        '95 wills/estates': '95 Wills/Estates',
        '95 wills and estates': '95 Wills/Estates',
        '96 advance directives/powers of attorney': '96 Advance Directives/Powers of Attorney',
        '96 advanced directives/powers of attorney': '96 Advance Directives/Powers of Attorney',
        '97 municipal legal needs': '97 Municipal Legal Needs',
        '99 other miscellaneous': '99 Other Miscellaneous'
    }
    
    # Try direct mapping first (most efficient)
    if normalized in standardization_map:
        return standardization_map[normalized]
    
    # Try regex patterns as fallback for unknown variations
    for pattern, standardized_code in legal_problem_patterns.items():
        if re.search(pattern, problem_str, re.IGNORECASE):
            return standardized_code
    
    # Final fallback: match by numeric code alone (catches completely unknown variations)
    if numeric_code:
        code_lookup = {
            '01': '01 Bankruptcy/Debtor Relief',
            '02': '02 Collection (including Repo/Def/Garnish)',
            '03': '03 Contracts/Warranties',
            '04': '04 Collection Practices/Creditor Harassment',
            '05': '05 Predatory Lending Practices (not mortgages)',
            '06': '06 Loans/Installment Purch.',
            '07': '07 Public Utilities',
            '08': '08 Unfair and Deceptive Sales and Practices (not real property)',
            '09': '09 Other Consumer/Finance',
            '12': '12 Discipline (including expulsion and suspension)',
            '13': '13 Special Education/Learning Disabilities',
            '14': '14 Access (Including Bilingual, Residency, Testing)',
            '15': '15 Vocational Education',
            '16': '16 Student Financial Aid',
            '19': '19 Other Education',
            '21': '21 Employment Discrimination',
            '22': '22 Wage Claim and other FLSA Issues',
            '23': '23 EITC (Earned Income Tax Credit)',
            '24': '24 Taxes (not EITC)',
            '25': '25 Employee Rights',
            '29': '29 Other Employment',
            '30': '30 Adoption',
            '31': '31 Custody/Visitation',
            '32': '32 Divorce/Sep./Annul.',
            '33': '33 Adult Guardianship/Conserv.',
            '34': '34 Name Change',
            '35': '35 Parental Rights Termin.',
            '36': '36 Paternity',
            '37': '37 Domestic Abuse',
            '38': '38 Support',
            '39': '39 Other Family',
            '41': '41 Delinquent',
            '42': '42 Neglected/Abused/Depend.',
            '43': '43 Emancipation',
            '44': '44 Minor Guardian/Conservatorship',
            '49': '49 Other Juvenile',
            '51': '51 Medicaid',
            '52': '52 Medicare',
            '53': "53 Government Children's Health Insurance Programs",
            '54': '54 Home and Community Based Care',
            '55': '55 Private Health Insurance',
            '56': '56 Long Term Health Care Facilities',
            '57': '57 State and Local Health',
            '59': '59 Other Health',
            '61': '61 Fed. Subsidized Housing',
            '62': '62 Homeownership/Real Prop. (not foreclosure)',
            '63': '63 Private Landlord/Tenant',
            '64': '64 Public Housing',
            '65': '65 Mobile Homes',
            '66': '66 Housing Discrimination',
            '67': '67 Mortgage Foreclosures (not predatory Lending/practices)',
            '68': '68 Mortgage Predatory Lending/Practices',
            '69': '69 Other Housing',
            '71': '71 TANF',
            '72': '72 Social Security (not SSDI)',
            '73': '73 Food Stamps',
            '74': '74 SSDI',
            '75': '75 SSI',
            '76': '76 Unemployment Compensation',
            '77': '77 Veterans Benefits',
            '78': '78 State and Local Income Maintenance',
            '79': '79 Other Income Maintenance',
            '81': '81 Immigration/Naturalization',
            '82': '82 Mental Health',
            '84': '84 Disability Rights',
            '85': '85 Civil Rights',
            '86': '86 Human Trafficking',
            '87': '87 Expungement',
            '89': '89 Other Individual Rights',
            '93': '93 Licenses (Auto and Other)',
            '94': '94 Torts',
            '95': '95 Wills/Estates',
            '96': '96 Advance Directives/Powers of Attorney',
            '97': '97 Municipal Legal Needs',
            '99': '99 Other Miscellaneous'
        }
        
        if numeric_code in code_lookup:
            return code_lookup[numeric_code]
    
    # If no match found, return original
    return problem_code

def clean_race_with_regex(race_value):
    """
    Clean race values using regex for flexible matching
    """
    if pd.isna(race_value) or str(race_value).strip() == '':
        return 'Other/Unknown'
    
    race_str = str(race_value).strip().lower()
    
    # White patterns
    if re.search(r'\b(white|caucasian)\b', race_str):
        return 'White'
    
    # Black patterns
    if re.search(r'\b(black|african.?american|aa)\b', race_str):
        return 'Black'
    
    # Native American patterns
    if re.search(r'\b(native|american.?indian|alaska.?native)\b', race_str):
        return 'Native American'
    
    # Asian/Pacific Islander patterns
    if re.search(r'\b(asian|pacific.?islander|hawaiian)\b', race_str):
        return 'Asian/Pacific Islander'
    
    # Hispanic patterns
    if re.search(r'\b(hispanic|latino|latina|latinx)\b', race_str):
        return 'Hispanic'
    
    # Multiracial patterns
    if re.search(r'\b(multi.?racial|multiracial|two.?or.?more)\b', race_str):
        return 'Multiracial'
    
    # Check for "and" which often indicates multiracial
    if ' and ' in race_str:
        return 'Multiracial'
    
    # Organization/Group
    if re.search(r'\b(organization|group)\b', race_str):
        return 'Other/Unknown'
    
    return 'Other/Unknown'

def clean_gender_with_regex(gender_value):
    """
    Clean gender values using regex for flexible matching
    """
    if pd.isna(gender_value) or str(gender_value).strip() == '':
        return 'Other/Unknown'
    
    gender_str = str(gender_value).strip().lower()
    
    # Female patterns
    if re.search(r'^(f|female|woman)$', gender_str):
        return 'Female'
    
    # Male patterns
    if re.search(r'^(m|male|man)$', gender_str):
        return 'Male'
    
    # Transgender patterns
    if re.search(r'\b(trans|transgender)\b', gender_str):
        return 'Transgender'
    
    # Non-binary patterns
    if re.search(r'\b(non.?binary|nonbinary|genderqueer|gender.?non.?conforming)\b', gender_str):
        return 'Non-binary'
    
    # Unknown/Other patterns
    if re.search(r'\b(don.?t.?know|unknown|prefer.?not|decline|other)\b', gender_str):
        return 'Other/Unknown'
    
    # Single letters or numbers that aren't F or M
    if re.match(r'^[a-eg-z0-9]$', gender_str):
        return 'Other/Unknown'
    
    return 'Other/Unknown'

# Standardization function
def standardize_new_data(df, upload_source):  
    column_mapping, race_mapping, gender_mapping, legal_problem_mapping = get_standard_mappings()
    
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
        # Step 1: Direct mapping
        df['race'] = df['race'].replace(race_mapping)
        # Step 2: Regex-based cleaning for anything not mapped
        df['race'] = df['race'].apply(
            lambda x: clean_race_with_regex(x) if x not in race_mapping.values() else x
        )

    # Standardize gender categories 
    if 'gender' in df.columns:
        # Step 1: Direct mapping
        df['gender'] = df['gender'].replace(gender_mapping)
        # Step 2: Regex-based cleaning for anything not mapped
        df['gender'] = df['gender'].apply(
            lambda x: clean_gender_with_regex(x) if x not in gender_mapping.values() else x
        )
        
    # Standardize legal problem codes
    if 'legal_problem_code' in df.columns:
        # Step 1: Clean whitespace
        df['legal_problem_code'] = df['legal_problem_code'].astype(str).str.strip()
        
        # Step 2: Apply mapping function
        df['legal_problem_code'] = df['legal_problem_code'].apply(
            lambda x: map_legal_problem_with_regex(x, legal_problem_mapping)
        )
        
        # Step 3: Final cleanup for any edge cases that slipped through
        final_cleanup = {
            '62 Homeownership/Real Property (not Foreclosure)': '62 Homeownership/Real Prop. (not foreclosure)',
            '62 Homeownership/Real Property (Not Foreclosure)': '62 Homeownership/Real Prop. (not foreclosure)',
            '08 Unfair and Deceptive Sales Practices (Not Real Property)': '08 Unfair and Deceptive Sales and Practices (not real property)',
            '67 Mortgage Foreclosures (Not Predatory Lending/Practices)': '67 Mortgage Foreclosures (not predatory Lending/practices)',
            '67 Mortgage Foreclosures (not Predatory Lending/Practices)': '67 Mortgage Foreclosures (not predatory Lending/practices)',
            '05 Predatory Lending Practices (Not Mortgages)': '05 Predatory Lending Practices (not mortgages)',
            '05 Predatory Lending Practices (not Mortgages)': '05 Predatory Lending Practices (not mortgages)',
        }
        df['legal_problem_code'] = df['legal_problem_code'].replace(final_cleanup)

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
        # Only convert Yes/No, leave everything else as-is (blanks will stay as empty strings)
        df['income_eligible'] = df['income_eligible'].apply(
            lambda x: x if x in ['Yes', 'No'] else ''
        )
    
    # Normalize asset_eligible 
    if 'asset_eligible' in df.columns:
        df['asset_eligible'] = df['asset_eligible'].astype(str).str.strip().str.capitalize()
        df['asset_eligible'] = df['asset_eligible'].apply(
            lambda x: x if x in ['Yes', 'No'] else ''
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
    
    # Convert numeric columns with special handling for outcome_amount
    numeric_cols = ['poverty_pct', 'adj_poverty_pct', 'age_intake', 'outcome_amount', 'case_time']
    for col in numeric_cols:
        if col in df.columns:
            if col == 'outcome_amount':
                # Special handling for currency format - remove $ and commas
                df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
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
        FILE_ID = "1Rj3Cwwc54vmqWk-TC4yie9i9fY6Vdphtphk11P5jj9o"
        
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
        
        # Convert numeric columns 
        numeric_columns = [
            'poverty_pct', 'adj_poverty_pct', 'age_intake', 'outcome_amount', 'case_time',
            'household_total', 'household_adults', 'household_children', 'days_open'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                if col == 'outcome_amount':
                    # Special handling for currency format
                    df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
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
                
        # Apply standardization to loaded data
        _, race_mapping, gender_mapping, _ = get_standard_mappings()
        
        # Clean race
        if 'race' in df.columns:
            df['race'] = df['race'].astype(str)  # Convert from categorical back to string temporarily
            df['race'] = df['race'].replace(race_mapping)
            df['race'] = df['race'].apply(
                lambda x: clean_race_with_regex(x) if pd.notna(x) and x not in race_mapping.values() else x
            )
            df['race'] = pd.Categorical(df['race'])  # Convert back to categorical for optimization
        
        # Clean gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str)  # Convert from categorical back to string temporarily
            df['gender'] = df['gender'].replace(gender_mapping)
            df['gender'] = df['gender'].apply(
                lambda x: clean_gender_with_regex(x) if pd.notna(x) and x not in gender_mapping.values() else x
            )
            df['gender'] = pd.Categorical(df['gender'])  # Convert back to categorical for optimization
            
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
        FILE_ID = "1Rj3Cwwc54vmqWk-TC4yie9i9fY6Vdphtphk11P5jj9o"
        
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

# create back up audit log
def create_backup_and_audit_log(username, upload_info):
    """
    Create a backup of current data and log the upload activity
    """
    try:
        creds_dict = get_google_credentials()
        if creds_dict is None:
            return False
        
        scopes = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        # Main data file ID
        MAIN_FILE_ID = "1Rj3Cwwc54vmqWk-TC4yie9i9fY6Vdphtphk11P5jj9o"
        
        # Backup file ID 
        BACKUP_FILE_ID = "1hJYhGULLWipg66ZTt93zlXvTLrdnXnM5KZ0-PbmHlD0"
        
        # Audit log file ID 
        AUDIT_LOG_FILE_ID = "1AzE8uz5hUzKoFE97xKzuSdIF9yDN4Y8kG54o5AenaxU"
        
        # 1. Create backup of current data
        main_file = gc.open_by_key(MAIN_FILE_ID)
        main_data = main_file.get_worksheet(0).get_all_values()
        
        backup_file = gc.open_by_key(BACKUP_FILE_ID)
        backup_worksheet = backup_file.get_worksheet(0)
        backup_worksheet.clear()
        backup_worksheet.update(main_data)
        
        # 2. Update audit log
        audit_file = gc.open_by_key(AUDIT_LOG_FILE_ID)
        audit_worksheet = audit_file.get_worksheet(0)
        
        # Check if audit log has headers
        try:
            existing_data = audit_worksheet.get_all_values()
            if not existing_data or existing_data[0] != ['Timestamp', 'Username', 'Action', 'Records_Added', 'Organization', 'Total_Records_After']:
                # Add headers if they don't exist
                audit_worksheet.clear()
                audit_worksheet.append_row(['Timestamp', 'Username', 'Action', 'Records_Added', 'Organization', 'Total_Records_After'])
        except:
            # Create headers if sheet is empty
            audit_worksheet.append_row(['Timestamp', 'Username', 'Action', 'Records_Added', 'Organization', 'Total_Records_After'])
        
        # Add new audit entry
        import datetime
        import pytz
        central = pytz.timezone('US/Central')
        timestamp = datetime.datetime.now(central).strftime("%Y-%m-%d %H:%M:%S CT")
        
        audit_row = [
            timestamp,
            username,
            "Data Upload", 
            upload_info['records_added'],
            upload_info['organization'],
            upload_info['total_records_after']
        ]
        
        audit_worksheet.append_row(audit_row)
        
        return True
        
    except Exception as e:
        st.error(f"Error creating backup/audit log: {str(e)}")
        return False

def load_audit_log():
    """
    Load and display audit log
    """
    try:
        creds_dict = get_google_credentials()
        if creds_dict is None:
            return None
        
        scopes = ['https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        AUDIT_LOG_FILE_ID = "1AzE8uz5hUzKoFE97xKzuSdIF9yDN4Y8kG54o5AenaxU"
        
        audit_file = gc.open_by_key(AUDIT_LOG_FILE_ID)
        audit_data = audit_file.get_worksheet(0).get_all_values()
        
        if len(audit_data) <= 1:  # Only headers or empty
            return pd.DataFrame(columns=['Timestamp', 'Username', 'Action', 'Records_Added', 'Organization', 'Total_Records_After'])
        
        headers = audit_data[0]
        rows = audit_data[1:]
        
        df = pd.DataFrame(rows, columns=headers)
        
        # FIX: Remove " CT" from timestamp strings before parsing
        df['Timestamp'] = df['Timestamp'].str.replace(' CT', '', regex=False)
        
        # Now parse the datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Sort by timestamp, most recent first
        df = df.sort_values('Timestamp', ascending=False)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading audit log: {str(e)}")
        return None

def process_single_file(uploaded_file, source):
    """
    Process a single uploaded file with standardization and validation
    Returns: (processed_df, success, error_message)
    """
    try:
        # Read the uploaded file
        df_new = pd.read_excel(uploaded_file, header=0)

        # Clean MALS case IDs by removing the "E" from the 3rd position
        if source == 'MALS':
            if 'Matter/Case ID' in df_new.columns:
                df_new['Matter/Case ID'] = df_new['Matter/Case ID'].astype(str).apply(
                    lambda x: x[:2] + x[3:] if len(x) > 3 and x[2] == 'E' else x
                )
            elif 'Case # ID' in df_new.columns:
                df_new['Case # ID'] = df_new['Case # ID'].astype(str).apply(
                    lambda x: x[:2] + x[3:] if len(x) > 3 and x[2] == 'E' else x
                )

        # Header validation check
        first_row_headers = df_new.columns.astype(str).str.strip()
        known_headers = get_standard_mappings()[0].keys()

        has_title_row = (
            (len(first_row_headers) <= 3 and any(first_row_headers.str.len() > 30)) or
            (first_row_headers.isin(known_headers).sum() < 3)
        )

        no_valid_columns = first_row_headers.isin(known_headers).sum() == 0

        if has_title_row or no_valid_columns:
            return None, False, "File appears to have header issues or missing column names"

        # Process data using existing standardization
        df_processed = standardize_new_data(df_new, source)
        
        return df_processed, True, None
        
    except Exception as e:
        return None, False, str(e)

def rebuild_dataset_from_files(uploaded_files, file_sources):
    """
    Rebuild the entire dataset from uploaded raw files
    """
    try:
        current_user = get_current_username()
        
        with st.spinner('Processing files and rebuilding dataset...'):
            combined_data = []
            processing_log = []
            
            # Process each uploaded file using shared logic
            for file in uploaded_files:
                source = file_sources[file.name]
                df_processed, success, error_msg = process_single_file(file, source)
                
                if success:
                    combined_data.append(df_processed)
                    processing_log.append(f"✅ {file.name}: {len(df_processed)} records processed as {source}")
                else:
                    processing_log.append(f"❌ {file.name}: Error - {error_msg}")
                    st.error(f"Error processing {file.name}: {error_msg}")
            
            if combined_data:
                # Combine all processed data
                final_dataset = pd.concat(combined_data, ignore_index=True)
                
                # Create backup of current data first
                backup_info = {
                    'records_added': len(final_dataset),
                    'organization': 'REBUILD_OPERATION',
                    'total_records_after': len(final_dataset)
                }
                
                backup_success = create_backup_and_audit_log(current_user, backup_info)
                
                if backup_success:
                    # Save the new dataset
                    if save_to_google_drive(final_dataset):
                        st.cache_data.clear()
                        
                        # Store results in session state instead of displaying immediately
                        st.session_state.rebuild_complete = True
                        st.session_state.rebuild_summary = processing_log
                        st.session_state.rebuild_total = len(final_dataset)
                        st.session_state.confirm_rebuild = False
                        
                        # Rerun to show the completion state
                        st.rerun()
                        
                    else:
                        st.error("Failed to save rebuilt dataset to Google Drive")
                else:
                    st.error("Failed to create backup before rebuild")
            else:
                st.error("No files were successfully processed")
                
    except Exception as e:
        st.error(f"Error during dataset rebuild: {str(e)}")

def get_current_username():
    """Get the current logged-in username"""
    return st.session_state.get('current_username', 'Unknown User')

def is_admin_user():
    """Check if the current user is admin"""
    return st.session_state.get('current_username', '') == 'admin'

# Updated handle_file_upload function
def handle_file_upload():
    # Display different content based on upload stage
    if st.session_state.upload_success:
        st.success("✅ Dataset successfully updated!")
        if st.button("Upload Another File", key="upload_another_file_btn"):
            # Reset all states
            st.session_state.upload_stage = 'initial'
            st.session_state.processed_data = None
            st.session_state.upload_success = False
            st.rerun()
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
                    # Use the shared processing function
                    df_processed, success, error_msg = process_single_file(uploaded_file, upload_source)
                    
                    if not success:
                        if "header" in error_msg.lower():
                            st.error(
                                "⚠️ It looks like the uploaded file either includes a header or is missing row with column names.\n\n"
                                "Please make sure:\n"
                                "- The first row contains column names (like 'Client ID', 'Date Opened')\n"
                                "- Any report titles or labels above the header row are removed"
                            )
                        else:
                            st.error(f"⚠️ {error_msg}")
                        return
                    
                    # Store in session state
                    st.session_state.processed_data = df_processed
                    st.session_state.upload_stage = 'review'
                    st.rerun()
                    
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
            if st.button("Confirm and Save", type="primary", key="confirm_save_btn"):
                # Get current username
                current_user = get_current_username()
                
                with st.spinner('Creating backup and saving data...'):
                    # Combine datasets
                    combined_df = pd.concat([existing_df, df_processed], ignore_index=True)
                    
                    # Prepare upload info for audit
                    upload_info = {
                        'records_added': len(df_processed),
                        'organization': df_processed['source'].iloc[0] if len(df_processed) > 0 else 'Unknown',
                        'total_records_after': len(combined_df)
                    }
                    
                    # Create backup and audit entry BEFORE saving new data
                    backup_success = create_backup_and_audit_log(current_user, upload_info)
                    
                    if not backup_success:
                        st.error("Failed to create backup. Upload cancelled for data safety.")
                    else:
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
                            st.rerun()
                        else:
                            st.error("Failed to save data to Google Drive. Please try again.")
            
            if st.button("Cancel", key="cancel_upload_btn"):
                st.session_state.upload_stage = 'initial'
                st.session_state.processed_data = None
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing existing data: {str(e)}")
            if st.button("Start Over", key="start_over_btn"):
                st.session_state.upload_stage = 'initial'
                st.session_state.processed_data = None
                st.rerun()

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
if st.sidebar.button('Refresh Data', key="refresh_data_btn"):
    # Clear Streamlit's cache
    st.cache_data.clear()
    st.session_state.data_loaded = False
    st.rerun()

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

# Data Cleaning Function for cleaner display
def clean_demographics_for_viz(df):
    """Clean demographic data specifically for visualizations"""
    df_clean = df.copy()
    
    # Fix age = 0 (treat as null/missing)
    df_clean['age_intake_clean'] = df_clean['age_intake'].replace(0, pd.NA)
    
    # Use the same standardization logic
    _, race_mapping, gender_mapping, _ = get_standard_mappings()
    
    # Standardize race
    df_clean['race_clean'] = df_clean['race'].replace(race_mapping)
    df_clean['race_clean'] = df_clean['race_clean'].apply(
        lambda x: clean_race_with_regex(x) if pd.notna(x) and x not in race_mapping.values() else x
    )
    
    # Standardize gender
    df_clean['gender_clean'] = df_clean['gender'].replace(gender_mapping)
    df_clean['gender_clean'] = df_clean['gender_clean'].apply(
        lambda x: clean_gender_with_regex(x) if pd.notna(x) and x not in gender_mapping.values() else x
    )
    
    return df_clean

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

    # Clean the data for visualization
    viz_df = clean_demographics_for_viz(filtered_df)

    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        st.subheader("Age Distribution")
        unique_age_dist = viz_df.groupby('client_id')['age_intake_clean'].first().dropna()
        fig = px.histogram(unique_age_dist, 
                  title="Age Distribution at Intake (Unique Clients)",
                  labels={'age_intake': 'Age at Intake', 'count': 'Number of Clients'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender distribution
        st.subheader("Gender Distribution")
        gender_counts = viz_df.groupby('client_id')['gender_clean'].first().value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index,
            title="Gender Distribution (Unique Clients)",
            labels={'names': 'Gender', 'values': 'Number of Clients'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Race distribution
        st.subheader("Race Distribution")
        race_counts = viz_df.groupby('client_id')['race_clean'].first().value_counts()
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
    problem_counts = display_df['legal_problem_code'].value_counts().head(20)
    
    fig = px.bar(x=problem_counts.index, y=problem_counts.values,
            title=f"Top 20 Legal Problems {'(Excluding Food Stamps)' if exclude_foodstamps else '(Including Food Stamps)'}",
            labels={'x': 'Legal Problem Code', 'y': 'Number of Cases'})
    fig.update_layout(
        xaxis_title="Legal Problem Code",
        yaxis_title="Number of Cases",
        height=800
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # County heat map
    st.subheader("Geographic Distribution")

    # Tennessee counties filter
    tn_only = st.checkbox("Show Tennessee counties only", value=False, key="tab3_tn_filter")

    # Volume selection radio button
    view_option = st.radio("Select View", [
        "High Volume (1000+ cases)",
        "Medium Volume (100-999 cases)",
        "Low Volume (<100 cases)"
    ], horizontal=True)

    # Calculate county counts from filtered data
    county_counts = display_df['county_dispute'].value_counts()
    
    # Filter to Tennessee counties if requested
    if tn_only:
        tn_counties = {
            'Anderson', 'Bedford', 'Benton', 'Bledsoe', 'Blount', 'Bradley', 'Campbell', 'Cannon', 'Carroll', 
            'Carter', 'Cheatham', 'Chester', 'Claiborne', 'Clay', 'Cocke', 'Coffee', 'Crockett', 'Cumberland', 
            'Davidson', 'Decatur', 'DeKalb', 'Dickson', 'Dyer', 'Fayette', 'Fentress', 'Franklin', 'Gibson', 
            'Giles', 'Grainger', 'Greene', 'Grundy', 'Hamblen', 'Hamilton', 'Hancock', 'Hardeman', 'Hardin', 
            'Hawkins', 'Haywood', 'Henderson', 'Henry', 'Hickman', 'Houston', 'Humphreys', 'Jackson', 'Jefferson', 
            'Johnson', 'Knox', 'Lake', 'Lauderdale', 'Lawrence', 'Lewis', 'Lincoln', 'Loudon', 'Macon', 'Madison', 
            'Marion', 'Marshall', 'Maury', 'McMinn', 'McNairy', 'Meigs', 'Monroe', 'Montgomery', 'Moore', 'Morgan', 
            'Obion', 'Overton', 'Perry', 'Pickett', 'Polk', 'Putnam', 'Rhea', 'Roane', 'Robertson', 'Rutherford', 
            'Scott', 'Sequatchie', 'Sevier', 'Shelby', 'Smith', 'Stewart', 'Sullivan', 'Sumner', 'Tipton', 'Trousdale', 
            'Unicoi', 'Union', 'Van Buren', 'Warren', 'Washington', 'Wayne', 'Weakley', 'White', 'Williamson', 'Wilson'
        }
        county_counts = county_counts[county_counts.index.isin(tn_counties)]

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
                        .head(20))
    
    fig = px.bar(top_problems_by_area, 
            x='count', 
            y='legal_problem_code',
            color='area_type',
            barmode='group',
            title='Top 20 Legal Problems by Urban/Rural Areas',
            labels={'count': 'Number of Cases', 
               'legal_problem_code': 'Legal Problem Code',
               'area_type': 'Area Type'})
    fig.update_layout(
        showlegend=True,
        xaxis_title="Number of Cases",
        yaxis_title="Legal Problem",
        height=800
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
        height=600,
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
        height=700,
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
        # First get unique client-problem combinations AND remove any blank/null legal problem codes
        unique_client_problems = df.dropna(subset=['client_id', 'legal_problem_code'])
        
        # Also filter out empty strings and whitespace-only entries
        unique_client_problems = unique_client_problems[
            (unique_client_problems['legal_problem_code'].str.strip() != '') &
            (unique_client_problems['legal_problem_code'].notna())
        ]
        
        # Remove duplicates
        unique_client_problems = unique_client_problems.drop_duplicates(['client_id', 'legal_problem_code'])
        
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
        st.info("📊 This analysis shows how often clients have multiple legal issues simultaneously (client-based analysis)")
        
        # Check if we have enough data
        if len(display_df) < 2:
            st.warning("Not enough data for co-occurrence analysis. Need at least 2 cases.")
            return
        
        # Check if we have the required columns
        if 'client_id' not in display_df.columns or 'legal_problem_code' not in display_df.columns:
            st.error("Missing required columns for co-occurrence analysis.")
            return
        
        # Filter out rows with missing values AND blank legal problem codes
        analysis_df = display_df.dropna(subset=['client_id', 'legal_problem_code'])
        analysis_df = analysis_df[
            (analysis_df['legal_problem_code'].str.strip() != '') &
            (analysis_df['legal_problem_code'].notna())
        ]
        
        if len(analysis_df) < 2:
            st.warning("Not enough valid data for co-occurrence analysis after removing missing values.")
            return
        
        try:
            # Calculate co-occurrence matrix
            cooccurrence_matrix, problem_frequencies = calculate_cooccurrence_matrix(analysis_df)
            
            if len(problem_frequencies) == 0:
                st.warning("No valid legal problem codes found for co-occurrence analysis.")
                return
            
            # Sort problems by their code number (natural order)
            sorted_problems = sorted(problem_frequencies.index.tolist())
            
            # Create a selectbox with CLIENT counts (not case counts)
            selected_problem = st.selectbox(
                "Select a Legal Problem to see co-occurring issues:",
                options=sorted_problems,
                format_func=lambda x: f"{x} ({problem_frequencies[x]:,} unique clients)"
            )
            
            if selected_problem:
                # Get co-occurrences for selected problem
                cooccurrences = cooccurrence_matrix[selected_problem].sort_values(ascending=False)
                
                # Remove self-reference and zero counts
                cooccurrences = cooccurrences[
                    (cooccurrences.index != selected_problem) & 
                    (cooccurrences > 0)
                ]
                
                if len(cooccurrences) == 0:
                    st.info(f"No co-occurring problems found for {selected_problem}. This means clients with this issue typically don't have other legal problems in the dataset.")
                    return
                
                # Calculate percentages
                total_clients = problem_frequencies[selected_problem]
                cooccurrence_pcts = (cooccurrences / total_clients * 100).round(1)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Co-occurring Problem': cooccurrences.index,
                    'Number of Clients with Both Issues': cooccurrences.values,
                    'Percentage of Clients with Both Issues': cooccurrence_pcts.values
                })
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Clients With Selected Issue", f"{total_clients:,}")
                with col2:
                    st.metric("Co-occurring Problems Found", f"{len(cooccurrences):,}")
                with col3:
                    if not results_df.empty:
                        most_common = results_df.iloc[0]['Co-occurring Problem']
                        most_common_pct = results_df.iloc[0]['Percentage of Clients with Both Issues']
                        st.metric("Most Common Co-occurrence", f"{most_common_pct:.1f}%")
                    else:
                        st.metric("Most Common Co-occurrence", "None")
                
                # Display results with clear explanation
                st.markdown(f"""
                #### Co-occurrence Analysis for: {selected_problem}
                
                **Analysis Type**: Client-based (each client counted once, regardless of number of cases)
                
                The table below shows other legal problems that appear for **the same clients** who have a {selected_problem} case.
                This helps identify patterns where clients face multiple types of legal issues.
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
                    file_name=f"client_cooccurrence_{selected_problem.replace('/', '_').replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"download_cooccurrence_{hash(selected_problem)}"  # Using hash to create unique key
                )
                                
        except Exception as e:
            st.error(f"Error in co-occurrence analysis: {str(e)}")
            st.info("This might be due to insufficient data or data formatting issues.")
    
    # Call the function
    display_cooccurrence_analysis(display_df)

with tab5:
    # Global Tennessee counties toggle
    tn_counties_only = st.checkbox(
        "Show Tennessee counties only", 
        value=False, 
        key="tab5_tn_counties_global"
    )

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

    # Apply Tennessee counties filter globally
    if tn_counties_only:
        tn_counties = {
            'Anderson', 'Bedford', 'Benton', 'Bledsoe', 'Blount', 'Bradley', 'Campbell', 'Cannon', 'Carroll', 
            'Carter', 'Cheatham', 'Chester', 'Claiborne', 'Clay', 'Cocke', 'Coffee', 'Crockett', 'Cumberland', 
            'Davidson', 'Decatur', 'DeKalb', 'Dickson', 'Dyer', 'Fayette', 'Fentress', 'Franklin', 'Gibson', 
            'Giles', 'Grainger', 'Greene', 'Grundy', 'Hamblen', 'Hamilton', 'Hancock', 'Hardeman', 'Hardin', 
            'Hawkins', 'Haywood', 'Henderson', 'Henry', 'Hickman', 'Houston', 'Humphreys', 'Jackson', 'Jefferson', 
            'Johnson', 'Knox', 'Lake', 'Lauderdale', 'Lawrence', 'Lewis', 'Lincoln', 'Loudon', 'Macon', 'Madison', 
            'Marion', 'Marshall', 'Maury', 'McMinn', 'McNairy', 'Meigs', 'Monroe', 'Montgomery', 'Moore', 'Morgan', 
            'Obion', 'Overton', 'Perry', 'Pickett', 'Polk', 'Putnam', 'Rhea', 'Roane', 'Robertson', 'Rutherford', 
            'Scott', 'Sequatchie', 'Sevier', 'Shelby', 'Smith', 'Stewart', 'Sullivan', 'Sumner', 'Tipton', 'Trousdale', 
            'Unicoi', 'Union', 'Van Buren', 'Warren', 'Washington', 'Wayne', 'Weakley', 'White', 'Williamson', 'Wilson'
        }
        display_df = display_df[
            (display_df['county_dispute'].isin(tn_counties)) | 
            (display_df['county_dispute'].isna())
        ]

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
        # Create a helper function for data cleaning 
        def clean_categorical_data(series, column_name=None):
            """Clean categorical data by removing nulls and numeric artifacts"""
            clean = series.astype(str)
            clean = clean.replace(['nan', '<NA>', 'None', ''], 'Other')
            clean = clean.dropna()
            # Remove numeric artifacts (like 1, 2, 1.0, etc.)
            numeric_mask = clean.str.match(r'^\d+\.?0*$', na=False)
            clean.loc[numeric_mask] = 'Other'
            
            if column_name == 'race':
                race_mapping = {
                    'Black or AA': 'Black',
                    'Multi-Racial': 'Multiracial', 
                    'American Indian or Alaska Native': 'Native American',
                    'Other/Unknown': 'Other',
                    'Other Ethnic Group': 'Other',
                    'Organization/Group': 'Other'
                }
                clean = clean.replace(race_mapping)
                valid_races = ['White', 'Black', 'Hispanic', 'Multiracial', 'Asian/Pacific Islander', 'Native American', 'Other']
                clean = clean.apply(lambda x: x if x in valid_races else 'Other')
                
            elif column_name == 'gender':
                gender_mapping = {
                    'Transgender Female to Male': 'Transgender',
                    'Transgender Male to Female': 'Transgender',
                    'Trans man': 'Transgender',
                    'Trans woman': 'Transgender',
                    'Non-Binary': 'Non-binary',
                    "Don't Know": 'Other',
                    'G': 'Other'
                }
                clean = clean.replace(gender_mapping)
                valid_genders = ['Female', 'Male', 'Transgender', 'Non-binary', 'Other']
                clean = clean.apply(lambda x: x if x in valid_genders else 'Other')
            
            return clean

        if basic_plot_type == "Bar Chart":
            category_col = st.selectbox("Select Category", options=safe_categorical_columns)
            clean_series = clean_categorical_data(display_df[category_col], category_col)
            
            value_counts = clean_series.value_counts()
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
            clean_series = clean_categorical_data(display_df[category_col])
            
            value_counts = clean_series.value_counts().head(10)
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
        
        if exclude_zeros:
            df_plot = df_plot[df_plot["outcome_amount"] > 0]
            st.write(f"After excluding zeros: {len(df_plot)} rows")
        
        if len(df_plot) == 0:
            st.error("❌ No valid outcome amount data found!")
            st.stop()

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
                # Clean both columns before grouping
                clean_df = display_df.copy()
                
                # Clean the category column
                clean_df[category] = clean_df[category].replace(['', ' ', 'nan', 'NaN', 'null', 'NULL', 'None'], pd.NA)
                clean_df[category] = clean_df[category].astype(str).replace('nan', pd.NA)
                clean_df = clean_df.dropna(subset=[category])
                # Remove numeric-looking entries
                clean_df = clean_df[~clean_df[category].str.match(r'^\d+\.?0*$', na=False)]
                
                count_df = clean_df.groupby(["legal_problem_code", category]).size().reset_index(name='count')
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
            # Handle currency data type for outcome_amount
            if numeric_col == 'outcome_amount':
                df_plot[numeric_col] = df_plot[numeric_col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('nan', '')
                df_plot[numeric_col] = df_plot[numeric_col].replace('', np.nan)
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
            household_adults = st.number_input("Adults in Household", min_value=1, max_value=10, value=1, key="dv_adults")
            household_children = st.number_input("Children in Household", min_value=0, max_value=10, value=1, key="dv_children")
            
            # Auto-calculate total household size
            household_total = household_adults + household_children
            st.info(f"**Total Household Size**: {household_total} (auto-calculated)")
            
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
        
        submitted = st.form_submit_button("Predict Risk", type="primary")

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
            household_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=1, key="ct_adults") 
            household_children = st.number_input("Number of Children", min_value=0, max_value=10, value=1, key="ct_children")
            
            # Auto-calculate total household size
            household_total = household_adults + household_children
            st.info(f"**Total Household Size**: {household_total} (auto-calculated)")
            
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
        
        submitted = st.form_submit_button("Predict Case Time", type="primary")

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
        
        # Load model and get prediction
        case_time_model = load_case_time_model()
        if case_time_model is None:
            result = {
                'predicted_hours': None,
                'complexity_category': "Error", 
                'resource_allocation': "Could not load prediction model"
            }
        else:
            # Pass the model directly to avoid import issues
            result = predict_case_time_with_model(client_data, case_time_model)
        
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
    # Check if user is admin
    if not is_admin_user():
        st.header("Data Management & Upload")
        st.error("🔒 Access Denied")
        st.warning("Only administrators can access the data upload functionality.")
        st.info("Please contact an administrator if you need to upload data.")
        
        # Show limited read-only functionality
        st.subheader("📊 Upload History (View Only)")
        if st.button("View Audit Log", key="view_audit_readonly_btn"):
            audit_df = load_audit_log()
            if audit_df is not None and len(audit_df) > 0:
                st.dataframe(
                    audit_df,
                    column_config={
                        "Timestamp": st.column_config.DatetimeColumn("Upload Time"),
                        "Username": "User",
                        "Records_Added": st.column_config.NumberColumn("New Records"),
                        "Total_Records_After": st.column_config.NumberColumn("Total After Upload")
                    },
                    hide_index=True
                )
            else:
                st.info("No upload history found.")
        st.stop()  # Exit the tab early
    
    # Admin-only content 
    st.header("Data Management & Upload")
    st.success(f"👑 Admin Access Granted - Welcome {get_current_username()}")

    # Check if rebuild just completed
    if st.session_state.get('rebuild_complete', False):
        st.success("✅ Dataset successfully rebuilt!")
        
        # Show processing summary
        st.subheader("Processing Summary")
        for log_entry in st.session_state.get('rebuild_summary', []):
            st.write(log_entry)
        
        st.metric("Total Records in New Dataset", st.session_state.get('rebuild_total', 0))
        
        # Clear and continue button
        if st.button("Clear and Continue", type="primary", key="clear_rebuild_btn"):
            # Clear all rebuild-related session state
            for key in ['rebuild_complete', 'rebuild_summary', 'rebuild_total', 'confirm_rebuild']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.stop()  # Don't show the rest of the tab until they click "Clear and Continue"
    
    # Add audit trail and backup management
    # Create columns for the management tools
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Upload History")
        if st.button("View Audit Log", key="view_audit_btn"):
            audit_df = load_audit_log()
            if audit_df is not None and len(audit_df) > 0:
                st.dataframe(
                    audit_df,
                    column_config={
                        "Timestamp": st.column_config.DatetimeColumn("Upload Time"),
                        "Username": "User",
                        "Records_Added": st.column_config.NumberColumn("New Records"),
                        "Total_Records_After": st.column_config.NumberColumn("Total After Upload")
                    },
                    hide_index=True
                )
            else:
                st.info("No upload history found.")
    
    with col2:
        st.subheader("🔄 Rebuild Dataset")
        st.warning("⚠️ This will completely replace the current dataset with newly uploaded files.")
        
        # File uploader that accepts multiple files
        uploaded_files = st.file_uploader(
            "Upload Raw Data Files (Excel format)",
            type="xlsx",
            accept_multiple_files=True,
            help="Upload all raw data files (e.g., WTLS 2023, WTLS 2024, LAS 2023, etc.)"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files selected")
            
            # Show file preview
            with st.expander("Preview Selected Files"):
                for file in uploaded_files:
                    st.write(f"• {file.name}")
            
            # Source assignment for each file
            st.subheader("Assign Organization Source")
            file_sources = {}
            
            for i, file in enumerate(uploaded_files):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{file.name}**")
                with col_b:
                    file_sources[file.name] = st.selectbox(
                        "Source",
                        options=["LAET", "LAS", "WTLS", "MALS"],
                        key=f"source_{i}"
                    )
            
            if st.button("🚨 REBUILD DATASET", type="primary", key="rebuild_dataset_btn"):
                if st.session_state.get('confirm_rebuild', False):
                    rebuild_dataset_from_files(uploaded_files, file_sources)
                else:
                    st.session_state.confirm_rebuild = True
                    st.error("⚠️ Click 'REBUILD DATASET' again to confirm. This will COMPLETELY REPLACE the current dataset!")
    
    st.markdown("---")
    
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
if st.sidebar.button("Prepare Excel Download", key="excel_download_btn"):
    with st.spinner('Preparing Excel file...'):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False)
        buffer.seek(0)
    
    # Move download button outside the spinner block to sidebar
    st.sidebar.download_button(
        label="Download Excel File",
        data=buffer,
        file_name="filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_btn"
    )

















