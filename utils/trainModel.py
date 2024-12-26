import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import joblib
import json
import os
from datetime import datetime
from config import Config

class ModelTrainer:
    def __init__(self):
        if not os.path.exists(Config.MODEL_VERSION_FILE):
            self._save_version_info({'version': 1, 'last_updated': None})
        
        self.female_model = LogisticRegression()
        self.male_model = GaussianNB()
    
    def train_models(self, include_collected=True):
        # Load and preprocess data
        df = pd.read_csv(Config.INITIAL_DATA_PATH)
        
        if include_collected and os.path.exists(Config.COLLECTED_DATA_PATH):
            collected_df = pd.read_csv(Config.COLLECTED_DATA_PATH)
            if not collected_df.empty:
                df = pd.concat([df, collected_df], ignore_index=True)
        
        # Train gender-specific models
        female_data = df[df['Gender'] == 'Female']
        male_data = df[df['Gender'] == 'Male']
        
        # Train female model
        X_f = female_data[['meanF0', 'stdevF0', 'rapJitter']]
        y_f = (female_data['Diagnosis'] == 'T2DM').astype(int)
        self.female_model.fit(X_f, y_f)
        
        # Train male model
        X_m = male_data[['meanInten', 'apq11Shimmer']]
        y_m = (male_data['Diagnosis'] == 'T2DM').astype(int)
        self.male_model.fit(X_m, y_m)
        
        self._save_models()
        
        # Return accuracy metrics
        return {
            'female_accuracy': self.female_model.score(X_f, y_f),
            'male_accuracy': self.male_model.score(X_m, y_m)
        }
    
    def _save_version_info(self, info):
        with open(Config.MODEL_VERSION_FILE, 'w') as f:
            json.dump(info, f)
    
    def _save_models(self):
        version_info = self._get_version_info()
        version_info['version'] += 1
        version_info['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        joblib.dump(self.female_model, 
                   os.path.join(Config.MODELS_DIR, f'female_model_v{version_info["version"]}.joblib'))
        joblib.dump(self.male_model, 
                   os.path.join(Config.MODELS_DIR, f'male_model_v{version_info["version"]}.joblib'))
        
        self._save_version_info(version_info)
    
    def _get_version_info(self):
        with open(Config.MODEL_VERSION_FILE, 'r') as f:
            return json.load(f)

def train_initial_models():
    """Function to train initial models"""
    trainer = ModelTrainer()
    accuracies = trainer.train_models(include_collected=False)
    print(f"Initial models trained with accuracies: {accuracies}")
    
def update_models():
    """Function to update models with collected data"""
    trainer = ModelTrainer()
    accuracies = trainer.train_models(include_collected=True)
    print(f"Models updated with accuracies: {accuracies}")