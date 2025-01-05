import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import logging
import joblib
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class T2DMPredictor:
    def __init__(self):
        """Initialize predictor with gender-specific models and scalers"""
        self.female_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        self.female_scaler = StandardScaler()
        self.male_model = GaussianNB()
        self.male_scaler = StandardScaler()
        self.female_imputer = SimpleImputer(strategy='mean')
        self.male_imputer = SimpleImputer(strategy='mean')
        self.models_dir = Config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

    def _preprocess_data(self, data_path):
        """Load and preprocess data from CSV file"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded dataset with {len(df)} entries")

            # Check for missing values
            missing_values = df.isnull().sum()
            logger.info(f"Missing values per column:\n{missing_values}")

            # Ensure required columns exist
            required_columns = ['meanF0', 'stdevF0', 'rapJitter', 'meanInten', 
                              'apq11Shimmer', 'Age', 'BMI', 'Gender', 'Diagnosis']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in dataset: {missing_columns}")

            # Clean data
            df = df.dropna(subset=['Gender', 'Diagnosis'])  # Must have these values
            df['Gender'] = df['Gender'].str.capitalize()
            df['Diagnosis'] = df['Diagnosis'].str.upper()
            
            logger.info(f"Data shape after cleaning: {df.shape}")

            # Split data by gender
            female_data = df[df['Gender'] == 'Female']
            male_data = df[df['Gender'] == 'Male']

            logger.info(f"Female samples: {len(female_data)}, Male samples: {len(male_data)}")

            return female_data, male_data

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train(self, data_path):
        """Train gender-specific models using the paper's methodology"""
        try:
            # Load and preprocess data
            female_data, male_data = self._preprocess_data(data_path)
            
            # Train female model
            logger.info("Training female model")
            X_f = female_data[['meanF0', 'stdevF0', 'rapJitter', 'Age', 'BMI']]
            y_f = (female_data['Diagnosis'] == 'T2DM').astype(int)
            
            # Handle missing values
            X_f_imputed = self.female_imputer.fit_transform(X_f)
            logger.info(f"Female data shape after imputation: {X_f_imputed.shape}")
            
            # Scale features
            X_f_scaled = self.female_scaler.fit_transform(X_f_imputed)
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_f_balanced, y_f_balanced = smote.fit_resample(X_f_scaled, y_f)
            logger.info(f"Female data shape after SMOTE: {X_f_balanced.shape}")
            
            # Train and evaluate female model
            self.female_model.fit(X_f_balanced, y_f_balanced)
            female_scores = self._evaluate_model(X_f_balanced, y_f_balanced, 
                                               self.female_model, "Female")

            # Train male model
            logger.info("Training male model")
            X_m = male_data[['meanInten', 'apq11Shimmer', 'Age', 'BMI']]
            y_m = (male_data['Diagnosis'] == 'T2DM').astype(int)
            
            # Handle missing values
            X_m_imputed = self.male_imputer.fit_transform(X_m)
            logger.info(f"Male data shape after imputation: {X_m_imputed.shape}")
            
            # Scale features
            X_m_scaled = self.male_scaler.fit_transform(X_m_imputed)
            
            # Apply SMOTE
            X_m_balanced, y_m_balanced = smote.fit_resample(X_m_scaled, y_m)
            logger.info(f"Male data shape after SMOTE: {X_m_balanced.shape}")
            
            # Train and evaluate male model
            self.male_model.fit(X_m_balanced, y_m_balanced)
            male_scores = self._evaluate_model(X_m_balanced, y_m_balanced, 
                                             self.male_model, "Male")

            # Save models and preprocessing components
            self._save_models()

            return {
                'female_accuracy': female_scores.mean(),
                'female_std': female_scores.std(),
                'male_accuracy': male_scores.mean(),
                'male_std': male_scores.std()
            }

        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def _evaluate_model(self, X, y, model, gender):
        """Evaluate model using 5-fold cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv)
        logger.info(f"{gender} model accuracy (5-fold CV): {scores.mean():.2f} ± {scores.std():.2f}")
        return scores

    def _save_models(self):
        """Save trained models and preprocessing components"""
        try:
            # Save models and components
            joblib.dump(self.female_model, os.path.join(self.models_dir, 'female_model.joblib'))
            joblib.dump(self.male_model, os.path.join(self.models_dir, 'male_model.joblib'))
            joblib.dump(self.female_scaler, os.path.join(self.models_dir, 'female_scaler.joblib'))
            joblib.dump(self.male_scaler, os.path.join(self.models_dir, 'male_scaler.joblib'))
            joblib.dump(self.female_imputer, os.path.join(self.models_dir, 'female_imputer.joblib'))
            joblib.dump(self.male_imputer, os.path.join(self.models_dir, 'male_imputer.joblib'))
            logger.info("Models and preprocessing components saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

def main():
    """Main function to train models"""
    try:
        predictor = T2DMPredictor()
        results = predictor.train(Config.INITIAL_DATA_PATH)
        logger.info(f"Training completed with accuracies: {results}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()