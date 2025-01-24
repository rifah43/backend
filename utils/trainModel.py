import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
       """Initialize gender-specific models and features"""
       # Define feature sets
       self.female_features = [
           'stdevF0', 'meanF0', 'rapJitter', 'meanInten',
           'localDbShimmer', 'apq5Shimmer', 'apq11Shimmer'
       ]
       
       self.male_features = [
           'meanInten', 'apq11Shimmer', 'stdevInten', 'ppq5Jitter',
           'localJitter', 'localDbShimmer', 'localShimmer'
       ]
       
       # Initialize models
       self.female_clf = VotingClassifier([
           ('lr', LogisticRegression(max_iter=1000)),
           ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
           ('nb', GaussianNB())
       ], voting='soft')
       
       self.male_clf = StackingClassifier([
           ('nb', GaussianNB()),
           ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
       ], final_estimator=LogisticRegression())
       
       # Initialize scalers
       self.female_scaler = StandardScaler()
       self.male_scaler = StandardScaler()
       
       self.models_dir = Config.MODELS_DIR
       os.makedirs(self.models_dir, exist_ok=True)

   def engineer_features(self, X):
       """Create engineered features from voice parameters"""
       X_eng = X.copy()
       
       # Feature ratios and combinations
       X_eng['pitch_intensity_ratio'] = X_eng['meanF0'] / X_eng['meanInten']
       X_eng['jitter_shimmer_ratio'] = X_eng['rapJitter'] / X_eng['localShimmer']
       X_eng['voice_irregularity'] = X_eng['stdevF0'] * X_eng['stdevInten']
       
       # Aggregated shimmer/jitter features
       X_eng['mean_shimmer'] = X_eng[['localShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer']].mean(axis=1)
       X_eng['mean_jitter'] = X_eng[['localJitter', 'rapJitter', 'ppq5Jitter']].mean(axis=1)
       
       return X_eng

   def _preprocess_data(self, data_path):
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} entries")

        # Split by gender and select only numeric columns
        female_data = df[df['Gender'].str.lower() == 'female']
        male_data = df[df['Gender'].str.lower() == 'male']
        
        logger.info(f"Female samples: {len(female_data)}, Male samples: {len(male_data)}")

        # Select only numeric features
        numeric_cols = ['meanF0', 'stdevF0', 'meanInten', 'stdevInten', 'HNR', 
                       'localShimmer', 'localDbShimmer', 'apq3Shimmer', 'apq5Shimmer',  
                       'apq11Shimmer', 'localJitter', 'rapJitter', 'ppq5Jitter',
                       'Age', 'BMI']

        X_f = female_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        X_m = male_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Fill missing values with mean
        X_f = X_f.fillna(X_f.mean())
        X_m = X_m.fillna(X_m.mean())

        # Engineer features
        for X in [X_f, X_m]:
            X['pitch_intensity_ratio'] = X['meanF0'] / X['meanInten']
            X['jitter_shimmer_ratio'] = X['rapJitter'] / X['localShimmer']
            X['voice_irregularity'] = X['stdevF0'] * X['stdevInten']
            X['mean_shimmer'] = X[['localShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer']].mean(axis=1)
            X['mean_jitter'] = X[['localJitter', 'rapJitter', 'ppq5Jitter']].mean(axis=1)
            X['Age_risk'] = X['Age'].apply(self._calculate_age_risk)
            X['BMI_risk'] = X['BMI'].apply(self._calculate_bmi_risk)

        # Drop original Age and BMI columns
        X_f = X_f.drop(['Age', 'BMI'], axis=1)
        X_m = X_m.drop(['Age', 'BMI'], axis=1)

        # Extract labels
        y_f = (female_data['Diagnosis'].str.upper() == 'T2DM').astype(int)
        y_m = (male_data['Diagnosis'].str.upper() == 'T2DM').astype(int)

        return (X_f, y_f), (X_m, y_m)

    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise
    
   def train(self, data_path):
       """Train gender-specific models"""
       (X_f, y_f), (X_m, y_m) = self._preprocess_data(data_path)
       
       # Train female model
       X_f_scaled = self.female_scaler.fit_transform(X_f)
       self.female_clf.fit(X_f_scaled, y_f)
       
       # Evaluate female model
       female_scores = self.evaluate_model(self.female_clf, X_f_scaled, y_f, "Female")
       
       # Train male model
       X_m_scaled = self.male_scaler.fit_transform(X_m)
       self.male_clf.fit(X_m_scaled, y_m)
       
       # Evaluate male model
       male_scores = self.evaluate_model(self.male_clf, X_m_scaled, y_m, "Male")
       
       self._save_models()
       
       return female_scores, male_scores

   def evaluate_model(self, model, X, y, gender):
       """Evaluate model using repeated cross-validation"""
       cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
       
       scores = cross_validate(model, X, y,
                             scoring=['accuracy', 'precision', 'recall', 'f1'],
                             cv=cv)
       
       logger.info(f"\n{gender} Model Performance:")
       logger.info(f"Accuracy: {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
       logger.info(f"Precision: {scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}")
       logger.info(f"Recall: {scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}")
       logger.info(f"F1: {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
       
       return scores

   def predict(self, features, gender):
       """Make prediction for single recording"""
       features_eng = self.engineer_features(pd.DataFrame([features]))
       
       # Add demographic risk scores
       features_eng['Age_risk'] = self._calculate_age_risk(features['Age'])
       features_eng['BMI_risk'] = self._calculate_bmi_risk(features['BMI'])
       
       if gender == 'Female':
           scaled_features = self.female_scaler.transform(features_eng)
           pred_proba = self.female_clf.predict_proba(scaled_features)[0][1]
           return pred_proba >= 0.54  # Female threshold from paper
       else:
           scaled_features = self.male_scaler.transform(features_eng)
           pred_proba = self.male_clf.predict_proba(scaled_features)[0][1]
           return pred_proba >= 0.46  # Male threshold from paper

   def _calculate_age_risk(self, age):
       """Calculate T2DM risk based on age"""
       if age < 40: return 0.1
       elif age < 50: return 0.2
       elif age < 60: return 0.3
       else: return 0.4

   def _calculate_bmi_risk(self, bmi):
       """Calculate T2DM risk based on BMI"""
       if bmi < 18.5: return 0.1
       elif bmi < 25: return 0.2
       elif bmi < 30: return 0.3
       elif bmi < 35: return 0.4
       else: return 0.5

   def _save_models(self):
       """Save trained models and scalers"""
       joblib.dump(self.female_clf, os.path.join(self.models_dir, 'female_model.joblib'))
       joblib.dump(self.male_clf, os.path.join(self.models_dir, 'male_model.joblib'))
       joblib.dump(self.female_scaler, os.path.join(self.models_dir, 'female_scaler.joblib'))
       joblib.dump(self.male_scaler, os.path.join(self.models_dir, 'male_scaler.joblib'))
       logger.info("Models saved successfully")

def main():
   predictor = EnhancedT2DMPredictor()
   female_scores, male_scores = predictor.train(Config.INITIAL_DATA_PATH)
   
   logger.info("\nTraining Complete")
   logger.info(f"Female Model Mean Accuracy: {female_scores['test_accuracy'].mean():.3f}")
   logger.info(f"Male Model Mean Accuracy: {male_scores['test_accuracy'].mean():.3f}")

if __name__ == "__main__":
   main()