import pandas as pd
import numpy as np

def preprocess_data(df):
    """Preprocess data for model training"""
    processed_df = df.copy()
    
    # Convert diagnosis to binary
    if 'Diagnosis' in processed_df.columns:
        processed_df['is_diabetic'] = (processed_df['Diagnosis'] == 'T2DM').astype(int)
    
    # Handle missing values
    numeric_columns = ['meanF0', 'stdevF0', 'rapJitter', 'meanInten', 'apq11Shimmer']
    processed_df[numeric_columns] = processed_df[numeric_columns].fillna(processed_df[numeric_columns].mean())
    
    # Normalize features
    for col in numeric_columns:
        mean = processed_df[col].mean()
        std = processed_df[col].std()
        processed_df[col] = (processed_df[col] - mean) / std
    
    return processed_df

def prepare_features_for_prediction(voice_data, gender):
    """Prepare features for model prediction"""
    if gender == 'Female':
        required_features = ['meanF0', 'stdevF0', 'rapJitter']
    else:
        required_features = ['meanInten', 'apq11Shimmer']
    
    features = {k: voice_data[k] for k in required_features}
    return features