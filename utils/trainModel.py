import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib
import os

class ModelTrainer:
    def __init__(self, data_path):
        """Initialize trainer with path to demo dataset"""
        self.data_path = data_path
        self.female_model = LogisticRegression()
        self.male_model = GaussianNB()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the demo dataset"""
        df = pd.read_csv(self.data_path)
        
        # Convert diagnosis to binary
        df['is_diabetic'] = (df['Diagnosis'] == 'T2DM').astype(int)
        
        # Split by gender
        self.female_data = df[df['Gender'] == 'Female']
        self.male_data = df[df['Gender'] == 'Male']
        
        # Prepare feature sets according to the paper
        self.female_features = self.female_data[['meanF0', 'stdevF0', 'rapJitter']]
        self.female_labels = self.female_data['is_diabetic']
        
        self.male_features = self.male_data[['meanInten', 'apq11Shimmer']]
        self.male_labels = self.male_data['is_diabetic']
        
    def train_models(self):
        """Train both gender-specific models"""
        # Train female model
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
            self.female_features, self.female_labels, test_size=0.2, random_state=42
        )
        self.female_model.fit(X_train_f, y_train_f)
        female_accuracy = self.female_model.score(X_test_f, y_test_f)
        
        # Train male model
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
            self.male_features, self.male_labels, test_size=0.2, random_state=42
        )
        self.male_model.fit(X_train_m, y_train_m)
        male_accuracy = self.male_model.score(X_test_m, y_test_m)
        
        return {
            'female_accuracy': female_accuracy,
            'male_accuracy': male_accuracy
        }
        
    def save_models(self, models_dir='models'):
        """Save trained models"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        joblib.dump(self.female_model, os.path.join(models_dir, 'female_model.joblib'))
        joblib.dump(self.male_model, os.path.join(models_dir, 'male_model.joblib'))