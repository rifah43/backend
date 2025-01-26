import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import logging
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedT2DMPredictor:
    def __init__(self):
        """Initialize gender-specific models and features."""
        self.voice_features = [
            'meanF0', 'stdevF0', 'meanInten', 'stdevInten', 'HNR',
            'localShimmer', 'localDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer',
            'localJitter', 'rapJitter', 'ppq5Jitter'
        ]
        
        # Models for each gender
        self.female_clf = LogisticRegression(max_iter=1000)  # Logistic Regression for females
        self.male_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest for males
        
        # Scalers
        self.female_scaler = StandardScaler()
        self.male_scaler = StandardScaler()

        # Feature selection for male data
        self.male_feature_selector = SelectKBest(score_func=f_classif, k=5)

        self.models_dir = Config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

    def _preprocess_data(self, data_path):
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded dataset with {len(df)} entries")

            # Split by gender
            female_data = df[df['Gender'].str.lower() == 'female']
            male_data = df[df['Gender'].str.lower() == 'male']
            
            logger.info(f"Female samples: {len(female_data)}, Male samples: {len(male_data)}")

            # Extract features and convert to numeric
            X_f = female_data[self.voice_features].apply(pd.to_numeric, errors='coerce')
            X_m = male_data[self.voice_features].apply(pd.to_numeric, errors='coerce')
            
            # Handle missing values
            X_f.fillna(X_f.mean(), inplace=True)
            X_m.fillna(X_m.mean(), inplace=True)

            # Add demographic features
            for X, data in [(X_f, female_data), (X_m, male_data)]:
                X['Age_risk'] = data['Age'].apply(self._calculate_age_risk)
                X['BMI_risk'] = data['BMI'].apply(self._calculate_bmi_risk)

            # Extract labels
            y_f = (female_data['Diagnosis'].str.upper() == 'T2DM').astype(int)
            y_m = (male_data['Diagnosis'].str.upper() == 'T2DM').astype(int)

            return (X_f, y_f), (X_m, y_m)

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train(self, data_path):
        """Train gender-specific models."""
        (X_f, y_f), (X_m, y_m) = self._preprocess_data(data_path)
        
        # Train female model
        X_f_scaled = self.female_scaler.fit_transform(X_f)
        self.female_clf.fit(X_f_scaled, y_f)
        
        # Train male model with feature selection
        X_m_selected = self.male_feature_selector.fit_transform(X_m, y_m)
        X_m_scaled = self.male_scaler.fit_transform(X_m_selected)
        self.male_clf.fit(X_m_scaled, y_m)
        
        # Evaluate models
        female_scores = self.evaluate_model(self.female_clf, X_f_scaled, y_f, "Female")
        male_scores = self.evaluate_model(self.male_clf, X_m_scaled, y_m, "Male")
        
        self._save_models()
        
        return female_scores, male_scores

    def evaluate_model(self, model, X, y, gender):
        """Evaluate model using repeated cross-validation."""
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        scores = cross_validate(
            model, X, y,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            cv=cv
        )
        
        logger.info(f"\n{gender} Model Performance:")
        logger.info(f"Accuracy: {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
        logger.info(f"Precision: {scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}")
        logger.info(f"Recall: {scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}")
        logger.info(f"F1 Score: {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
        
        return scores

    def _calculate_age_risk(self, age):
        if age < 40: return 0.01
        elif age < 50: return 0.012
        elif age < 60: return 0.015
        else: return 0.02

    def _calculate_bmi_risk(self, bmi):
        if bmi < 18.5: return 0.01
        elif bmi < 25: return 0.013
        elif bmi < 30: return 0.015
        elif bmi < 35: return 0.018
        else: return 0.02

    def _save_models(self):
        """Save trained models and scalers."""
        joblib.dump(self.female_clf, os.path.join(self.models_dir, 'female_model.joblib'))
        joblib.dump(self.male_clf, os.path.join(self.models_dir, 'male_model.joblib'))
        joblib.dump(self.female_scaler, os.path.join(self.models_dir, 'female_scaler.joblib'))
        joblib.dump(self.male_scaler, os.path.join(self.models_dir, 'male_scaler.joblib'))
        joblib.dump(self.male_feature_selector, os.path.join(self.models_dir, 'male_feature_selector.joblib'))
        logger.info("Models saved successfully.")

def main():
    predictor = EnhancedT2DMPredictor()
    female_scores, male_scores = predictor.train(Config.INITIAL_DATA_PATH)
    
    logger.info("\nTraining Complete")
    logger.info(f"Female Model Mean Accuracy: {female_scores['test_accuracy'].mean():.3f}")
    logger.info(f"Male Model Mean Accuracy: {male_scores['test_accuracy'].mean():.3f}")

if __name__ == "__main__":
    main()
