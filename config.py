import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    DATASET_DIR = BASE_DIR / 'dataset'
    MODELS_DIR = BASE_DIR / 'models'
    TEMP_DIR = BASE_DIR / 'temp'
    COLLECTED_DATA_PATH = DATASET_DIR / 'collected_data.csv'
    INITIAL_DATA_PATH = DATASET_DIR / 'initial_data.csv'
    MODEL_VERSION_FILE = MODELS_DIR / 'version_info.json'
    
    # Create directories if they don't exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)