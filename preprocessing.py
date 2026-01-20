"""
Preprocessing module for legal aid prediction models.
Contains functions for both domestic violence risk prediction and case time prediction.
"""

import pandas as pd
import numpy as np

# --- Domestic Violence Model Functions ---

def preprocess_client_data(client_data):
    """
    Prepares client data for domestic violence risk prediction.
    
    Parameters:
    -----------
    client_data : dict or pd.DataFrame
        A dictionary or DataFrame containing client information
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for model prediction
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(client_data, dict):
        df = pd.DataFrame([client_data])
    else:
        df = client_data.copy()
    
    # Calculate single_parent feature
    df['single_parent'] = ((df['household_adults'] == 1) & 
                         (df['household_children'] > 0)).astype(int)
    
    # Ensure zip_code is present (required by DV model)
    if 'zip_code' not in df.columns:
        df['zip_code'] = np.nan
    
    # Important: When using this in production, missing values will be handled
    # by the preprocessing pipeline inside the saved model
    
    return df

def interpret_risk_score(risk_score):
    """
    Updated thresholds for ROC AUC optimized model
    """
    if risk_score < 0.4:  # Adjusted for ROC AUC model
        risk_level = "Low"
        recommendation = "Standard intake process. Low probability of domestic violence based on intake data."
    elif risk_score < 0.7:  # Adjusted middle range  
        risk_level = "Medium"
        recommendation = "Consider additional screening questions during intake. Some risk factors present."
    else:
        risk_level = "High"
        recommendation = "Case shows risk factors similar to past DV cases. Recommend additional screening and consider connecting to resources."
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'recommendation': recommendation
    }

# --- Case Time Prediction Functions ---

def engineer_case_time_features(df):
    """
    Apply the exact same feature engineering as used in model training.
    This must match the engineer_features function from the training code.
    
    IMPORTANT: For single-row predictions, we use fallback values instead of .median()
    """
    data = df.copy()
    
    # Define fallback values for single-row predictions
    # These are reasonable defaults based on the training data
    AGE_FALLBACK = 45.0  # Middle-aged adult
    
    # Adult-to-child ratio (handle division by zero)
    data['adult_child_ratio'] = np.where(
        data['household_children'] == 0, 
        data['household_adults'], 
        data['household_adults'] / data['household_children']
    )
    
    # Household density (handle division by zero)
    data['household_density'] = np.where(
        data['household_adults'] == 0, 
        0, 
        data['household_total'] / data['household_adults']
    )
    
    # Poverty intensity (handle missing values)
    data['poverty_intensity'] = np.abs(data['adj_poverty_pct'].fillna(100) - 100)
    
    # Age groups (handle missing values and outliers)
    # Use fillna with a scalar value instead of .median() for single-row compatibility
    if data['age_intake'].isna().any():
        data['age_intake'] = data['age_intake'].fillna(AGE_FALLBACK)
    
    data['age_intake'] = np.clip(data['age_intake'], 18, 100)  # Reasonable age bounds
    data['age_group'] = pd.cut(data['age_intake'], 
                              bins=[0, 25, 45, 65, 100], 
                              labels=['young', 'middle', 'senior', 'elderly'])
    
    # County match (handle missing values)
    data['county_match'] = (
        data['county_residence'].fillna('unknown') == 
        data['county_dispute'].fillna('unknown')
    ).astype(int)
    
    # Group legal problem codes by major categories (exactly as in training)
    def group_legal_codes(code):
        if pd.isna(code):
            return 'unknown'
        code_str = str(code).strip()
        
        # Extract numeric prefix to determine category
        if code_str.startswith('0') or code_str.startswith('1'):
            return 'consumer_finance'  # 01-09: Bankruptcy, Collections, Contracts, etc.
        elif code_str.startswith('12') or code_str.startswith('13') or code_str.startswith('14') or code_str.startswith('16') or code_str.startswith('19'):
            return 'education'  # 12-19: Education issues
        elif code_str.startswith('2'):
            return 'employment'  # 21-29: Employment and tax issues
        elif code_str.startswith('3'):
            return 'family'  # 30-39: Family law matters
        elif code_str.startswith('4'):
            return 'juvenile'  # 41-49: Juvenile issues
        elif code_str.startswith('5'):
            return 'health'  # 51-59: Health and medical
        elif code_str.startswith('6'):
            return 'housing'  # 61-69: Housing and real estate
        elif code_str.startswith('7'):
            return 'income_benefits'  # 71-79: Government benefits
        elif code_str.startswith('8'):
            return 'civil_rights'  # 81-89: Individual rights and civil matters
        elif code_str.startswith('9'):
            return 'miscellaneous'  # 93-99: Licenses, estates, torts, etc.
        else:
            return 'other'
    
    # Additional interaction features for better predictions
    data['age_poverty_interaction'] = data['age_intake'] * data['poverty_pct'] / 100
    data['household_complexity'] = data['household_total'] * data['adult_child_ratio']
    
    # High-risk case indicators
    data['high_poverty'] = (data['adj_poverty_pct'] < 50).astype(int)  # Deep poverty
    data['elderly_case'] = (data['age_intake'] >= 65).astype(int)
    data['large_household'] = (data['household_total'] >= 5).astype(int)
    
    data['legal_problem_group'] = data['legal_problem_code'].apply(group_legal_codes)
    
    # Replace any remaining inf/nan values in engineered features
    # Use scalar fallbacks instead of .median() for single-row compatibility
    engineered_cols = ['adult_child_ratio', 'household_density', 'poverty_intensity', 'county_match',
                      'age_poverty_interaction', 'household_complexity', 'high_poverty', 
                      'elderly_case', 'large_household']
    
    fallback_values = {
        'adult_child_ratio': 2.0,
        'household_density': 1.5,
        'poverty_intensity': 50.0,
        'county_match': 0,
        'age_poverty_interaction': 45.0,
        'household_complexity': 3.0,
        'high_poverty': 0,
        'elderly_case': 0,
        'large_household': 0
    }
    
    for col in engineered_cols:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        if data[col].isna().any():
            data[col] = data[col].fillna(fallback_values.get(col, 0))
    
    return data

def preprocess_case_time_data(client_data):
    """
    Prepares client data for case time prediction with full feature engineering.
    
    Parameters:
    -----------
    client_data : dict or pd.DataFrame
        A dictionary or DataFrame containing client information
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for model prediction with features in correct order
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(client_data, dict):
        df = pd.DataFrame([client_data])
    else:
        df = client_data.copy()
    
    # Apply feature engineering (this is critical!)
    processed_data = engineer_case_time_features(df)
    
    # Ensure all required columns exist - the model expects these exact features
    # Base numerical features
    base_numerical = [
        'age_intake', 'household_total', 'household_adults', 
        'household_children', 'poverty_pct', 'adj_poverty_pct'
    ]
    
    # Engineered numerical features
    engineered_numerical = [
        'adult_child_ratio', 'household_density', 'poverty_intensity', 'county_match',
        'age_poverty_interaction', 'household_complexity', 'high_poverty', 
        'elderly_case', 'large_household'
    ]
    
    # Categorical features
    categorical_features = [
        'gender', 'race', 'disabled', 'veteran', 'county_residence',
        'county_dispute', 'living_arrangement', 'source', 
        'legal_problem_group', 'age_group'
    ]
    
    # Ensure all features are present
    all_features = base_numerical + engineered_numerical + categorical_features
    for feature in all_features:
        if feature not in processed_data.columns:
            processed_data[feature] = np.nan
    
    # Return only the features in the correct order
    return processed_data[all_features]

def interpret_case_time(predicted_time):
    """
    Interprets the predicted case time and provides resource allocation recommendations.
    
    Parameters:
    -----------
    predicted_time : float
        Predicted case time in hours
    
    Returns:
    --------
    dict
        Dictionary containing categorization and resource recommendations
    """
    # Round to 1 decimal place for cleaner display
    hours = round(predicted_time, 1)
    
    if hours < 3:
        category = "Brief Service"
        allocation = "This case is likely to require minimal resources."
    elif hours < 10:
        category = "Moderate Complexity"
        allocation = "This case will require moderate resources."
    else:
        category = "High Complexity"
        allocation = "This case is likely to require significant resources."
    
    return {
        'predicted_hours': hours,
        'complexity_category': category,
        'resource_allocation': allocation
    }

# --- Prediction Functions ---

def predict_domestic_violence_risk(client_data):
    """
    Predicts domestic violence risk for a client based on intake information.
    
    Parameters:
    -----------
    client_data : dict or pd.DataFrame
        Dictionary or DataFrame containing client information
    
    Returns:
    --------
    dict
        Dictionary containing risk score, level, and recommendation
    """
    try:
        # Load the model from Google Drive
        from app import load_dv_model
        model = load_dv_model()
        
        # Check if model loaded successfully
        if model is None:
            return {
                'risk_score': None,
                'risk_level': "Error",
                'recommendation': "Could not load prediction model"
            }
        
        # Preprocess data
        processed_data = preprocess_client_data(client_data)
        
        # Make prediction
        risk_score = model.predict_proba(processed_data)[0, 1]
        
        # Interpret result
        result = interpret_risk_score(risk_score)
        
        return result
    except Exception as e:
        print(f"Error predicting domestic violence risk: {e}")
        return {
            'risk_score': None,
            'risk_level': "Error",
            'recommendation': f"Could not process prediction: {str(e)}"
        }

def predict_case_time(client_data):
    """
    Predicts case time based on client intake information.
    Note: This version requires external model loading to avoid import conflicts.
    """
    return {
        'predicted_hours': None,
        'complexity_category': "Error",
        'resource_allocation': "Use predict_case_time_with_model instead to avoid import conflicts"
    }

def predict_case_time_with_model(client_data, model):
    """
    Predicts case time with a pre-loaded model (avoids import issues).
    """
    try:
        # Check if model loaded successfully
        if model is None:
            return {
                'predicted_hours': None,
                'complexity_category': "Error",
                'resource_allocation': "Could not load prediction model"
            }
        
        # Preprocess data with feature engineering
        processed_data = preprocess_case_time_data(client_data)
        
        # Make prediction
        predicted_time = model.predict(processed_data)[0]
        
        # Interpret result
        result = interpret_case_time(predicted_time)
        
        return result
    except Exception as e:
        print(f"Error in predict_case_time_with_model: {e}")
        import traceback
        traceback.print_exc()
        return {
            'predicted_hours': None,
            'complexity_category': "Error",
            'resource_allocation': f"Could not process prediction: {str(e)}"
        }
