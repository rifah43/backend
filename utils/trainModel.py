import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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

class T2DMVoicePredictor:
    def __init__(self):
        """Initialize models and scalers as per the research paper."""
        self.female_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        self.male_model = GaussianNB()
        self.female_scaler = StandardScaler()
        self.male_scaler = StandardScaler()
        self.models_dir = Config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

    def match_samples(self, data):
        """Match samples by age and BMI as per the research paper methodology."""
        matched_data = []
        try:
            t2dm_samples = data[data['Diagnosis'] == 'T2DM']
            non_diabetic = data[data['Diagnosis'] == 'Non-T2DM']

            for _, t2dm_sample in t2dm_samples.iterrows():
                matches = non_diabetic[
                    (abs(non_diabetic['Age'] - t2dm_sample['Age']) <= 15) &  # Relaxed criteria
                    (abs(non_diabetic['BMI'] - t2dm_sample['BMI']) <= 10)    # Relaxed criteria
                ]
                logger.info(f"T2DM Sample: Age={t2dm_sample['Age']}, BMI={t2dm_sample['BMI']}")
                logger.info(f"Matches found: {len(matches)}")
                if not matches.empty:
                    matched_data.append(t2dm_sample)
                    matched_data.append(matches.iloc[0])

            if not matched_data:
                logger.warning("No matches found.")
                return pd.DataFrame()

            return pd.DataFrame(matched_data)

        except Exception as e:
            logger.error(f"Error in matching samples: {str(e)}")
            raise


    def train(self, data_path):
        """Train gender-specific models using the paper's methodology."""
        try:
            # Load and standardize dataset
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded with {len(df)} entries.")

            df.rename(columns={
                'mean_f0': 'meanF0',
                'std_f0': 'stdevF0',
                'rap_jitter': 'rapJitter',
                'bmi': 'BMI'
            }, inplace=True)

            required_columns = ['meanF0', 'stdevF0', 'rapJitter', 'meanInten', 'apq11Shimmer', 'Age', 'BMI', 'Diagnosis', 'Gender']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns in dataset: {missing_columns}")
                raise ValueError(f"Missing columns: {missing_columns}")

            # Match samples by gender
            # Match samples by gender
            female_data = self.match_samples(df[df['Gender'] == 'Female'])
            male_data = self.match_samples(df[df['Gender'] == 'Male'])

            if female_data.empty:
                logger.warning("No matched samples for females. Using all available female data.")
                female_data = df[df['Gender'] == 'Female']

            if male_data.empty:
                logger.warning("No matched samples for males. Using all available male data.")
                male_data = df[df['Gender'] == 'Male']


            # Train female model
            logger.info("Training female model")
            X_f = female_data[['meanF0', 'stdevF0', 'rapJitter', 'Age', 'BMI']]
            y_f = (female_data['Diagnosis'] == 'T2DM').astype(int)
            X_f_scaled = self.female_scaler.fit_transform(X_f)
            smote = SMOTE(random_state=42)
            X_f_balanced, y_f_balanced = smote.fit_resample(X_f_scaled, y_f)
            
            # Tune female model hyperparameters
            param_grid = {'C': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000, 1500]}
            grid = GridSearchCV(LogisticRegression(class_weight='balanced', random_state=42), param_grid, cv=5)
            grid.fit(X_f_balanced, y_f_balanced)
            self.female_model = grid.best_estimator_

            # Train male model
            logger.info("Training male model")
            X_m = male_data[['meanInten', 'apq11Shimmer', 'Age', 'BMI']]
            y_m = (male_data['Diagnosis'] == 'T2DM').astype(int)
            X_m_scaled = self.male_scaler.fit_transform(X_m)
            X_m_balanced, y_m_balanced = smote.fit_resample(X_m_scaled, y_m)
            self.male_model.fit(X_m_balanced, y_m_balanced)

            # Evaluate models
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            female_scores = self._evaluate_model(X_f_balanced, y_f_balanced, self.female_model, cv, "Female")
            male_scores = self._evaluate_model(X_m_balanced, y_m_balanced, self.male_model, cv, "Male")

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

    def _evaluate_model(self, X, y, model, cv, gender):
        """Evaluate model using 5-fold cross-validation as per paper."""
        scores = cross_val_score(model, X, y, cv=cv)
        logger.info(f"{gender} model accuracy (5-fold CV): {scores.mean():.2f} ± {scores.std():.2f}")
        return scores

    def _save_models(self):
        """Save trained models and scalers."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            joblib.dump(self.female_model, os.path.join(self.models_dir, f'female_model_{timestamp}.joblib'))
            joblib.dump(self.male_model, os.path.join(self.models_dir, f'male_model_{timestamp}.joblib'))
            joblib.dump(self.female_scaler, os.path.join(self.models_dir, f'female_scaler_{timestamp}.joblib'))
            joblib.dump(self.male_scaler, os.path.join(self.models_dir, f'male_scaler_{timestamp}.joblib'))
            logger.info(f"Models and scalers saved with timestamp {timestamp}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

def main():
    """Main function to train models."""
    try:
        predictor = T2DMVoicePredictor()
        results = predictor.train(Config.INITIAL_DATA_PATH)
        logger.info(f"Training completed with accuracies: {results}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
