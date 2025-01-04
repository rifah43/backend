from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime
from utils.audioProcessing import extract_audio_features
# from models.trendData import TrendData
# from db import db
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

predict_bp = Blueprint('predict', __name__)

class Predictor:
    def __init__(self):
        """Initialize predictor with pre-trained models"""
        try:
            self.female_model = joblib.load(os.path.join(Config.MODELS_DIR, 'female_model.joblib'))
            self.male_model = joblib.load(os.path.join(Config.MODELS_DIR, 'male_model.joblib'))
            
            # Log model information for debugging
            print("Female model features:", self.female_model.n_features_in_)
            print("Male model features:", self.male_model.n_features_in_)
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def predict(self, voice_features, gender, age, bmi):
        """Make T2DM risk prediction based on voice features, gender, age and BMI"""
        try:
            # Extract voice features based on gender
            if gender.lower() == 'female':
                # For females: meanF0, stdevF0, age, bmi
                voice_array = np.array([[
                    voice_features['meanF0'],
                    voice_features['stdevF0'],
                    age,
                    bmi
                ]])
                model = self.female_model
            elif gender.lower() == 'male':
                # For males: meanInten, apq11Shimmer, age, bmi
                voice_array = np.array([[
                    voice_features['meanInten'],
                    voice_features['apq11Shimmer'],
                    age,
                    bmi
                ]])
                model = self.male_model
            else:
                raise ValueError("Invalid gender specified")

            # Log the feature array for debugging
            print(f"Input feature array shape: {voice_array.shape}")
            print(f"Input features: {voice_array}")
            
            # Get base probability from voice features
            voice_probability = model.predict_proba(voice_array)[0][1]
            
            # Adjust probability based on age and BMI risk factors
            age_factor = self._calculate_age_risk(age)
            bmi_factor = self._calculate_bmi_risk(bmi)
            
            # Combine probabilities with weights
            final_probability = (0.6 * voice_probability + 
                               0.2 * age_factor + 
                               0.2 * bmi_factor)
            
            return final_probability
            
        except Exception as e:
            print(f"Prediction error details: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def _calculate_age_risk(self, age):
        """Calculate risk factor based on age"""
        if age < 40:
            return 0.2
        elif age < 50:
            return 0.4
        elif age < 60:
            return 0.6
        else:
            return 0.8

    def _calculate_bmi_risk(self, bmi):
        """Calculate risk factor based on BMI"""
        if bmi < 18.5:
            return 0.3  # Underweight
        elif bmi < 25:
            return 0.2  # Normal weight
        elif bmi < 30:
            return 0.5  # Overweight
        else:
            return 0.8  # Obese

    def _store_prediction(self, features, gender, age, bmi, probability):
        """Store prediction data for future model updates"""
        try:
            data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'gender': gender,
                'age': age,
                'bmi': bmi,
                'prediction': probability,
                **features
            }
            df = pd.DataFrame([data])
            
            
            df.to_csv(
                Config.COLLECTED_DATA_PATH, 
                mode='a', 
                header=not os.path.exists(Config.COLLECTED_DATA_PATH), 
                index=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to store prediction: {str(e)}")

# Initialize predictor
predictor = Predictor()

@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making T2DM risk predictions from voice recordings with demographic data"""
    try:
        # Validate required fields
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        # Get and validate form data
        required_fields = ['user_id', 'gender', 'age', 'bmi']
        form_data = {}
        for field in required_fields:
            if field not in request.form:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            form_data[field] = request.form.get(field)
        
        # Validate numeric fields
        try:
            age = float(form_data['age'])
            bmi = float(form_data['bmi'])
            if age <= 0 or age > 120:
                return jsonify({'error': 'Invalid age value'}), 400
            if bmi <= 0 or bmi > 100:
                return jsonify({'error': 'Invalid BMI value'}), 400
        except ValueError:
            return jsonify({'error': 'Age and BMI must be numeric values'}), 400
            
        # Process audio file
        audio_file = request.files['audio']
        print(f"Processing audio file: {audio_file.filename}")
        
        try:
            features = extract_audio_features(audio_file, form_data['gender'])
            if not features:
                return jsonify({'error': 'Failed to extract voice features'}), 400
            print(f"Extracted features: {features}")
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return jsonify({'error': f'Failed to extract voice features: {str(e)}'}), 400
        
        # Make prediction
        try:
            risk_probability = predictor.predict(
                features, 
                form_data['gender'],
                age,
                bmi
            )
            print(f"Prediction result: {risk_probability}")
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Store trend data
        # try:
        #     trend_data = TrendData(
        #         user_id=form_data['user_id'],
        #         risk_probability=float(risk_probability),
        #         age=age,
        #         bmi=bmi,
        #         gender=form_data['gender'],
        #         **features
        #     )
        #     db.session.add(trend_data)
        #     db.session.commit()
        # except Exception as e:
        #     print(f"Database error: {str(e)}")
        #     db.session.rollback()
            # Don't return here - still send the prediction result to client
        

        return jsonify({
            'risk_probability': float(risk_probability),
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500