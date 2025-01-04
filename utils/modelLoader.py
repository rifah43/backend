import joblib
import json
from config import Config
import os

def load_latest_models():
    """Load the latest version of trained models"""
    with open(Config.MODEL_VERSION_FILE, 'r') as f:
        version_info = json.load(f)
    
    current_version = version_info['version']
    
    female_model = joblib.load(
        os.path.join(Config.MODELS_DIR, f'female_model_v{current_version}.joblib')
    )
    male_model = joblib.load(
        os.path.join(Config.MODELS_DIR, f'male_model_v{current_version}.joblib')
    )
    
    return female_model, male_model