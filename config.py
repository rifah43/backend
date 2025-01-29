import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    DATASET_DIR = BASE_DIR / 'dataset'
    MODELS_DIR = BASE_DIR / 'models'
    TEMP_DIR = BASE_DIR / 'temp'
    COLLECTED_DATA_PATH = DATASET_DIR / 'newdataset.csv'
    INITIAL_DATA_PATH = DATASET_DIR / 'dataset.csv'
    MODEL_VERSION_FILE = MODELS_DIR / 'version_info.json'
    RANDOM_STATE = 443
    FEMALE_THRESHOLD = 0.35
    MALE_THRESHOLD = 0.40
    
    # Age-adjusted thresholds (optional)
    THRESHOLD_AGE_ADJUSTMENT = {
        'under_40': {'female': 0.40, 'male': 0.45},
        '40_to_60': {'female': 0.35, 'male': 0.40},
        'over_60': {'female': 0.30, 'male': 0.35}
    }
    
    # Create directories if they don't exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)