from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime
from utils.audioProcessing import extract_audio_features
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

predict_bp = Blueprint('predict', __name__)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        """Initialize predictor with pre-trained models"""
        try:
            # Load models
            self.female_model = joblib.load(os.path.join(Config.MODELS_DIR, 'female_model.joblib'))
            self.male_model = joblib.load(os.path.join(Config.MODELS_DIR, 'male_model.joblib'))
            
            # Load scalers
            self.female_scaler = joblib.load(os.path.join(Config.MODELS_DIR, 'female_scaler.joblib'))
            self.male_scaler = joblib.load(os.path.join(Config.MODELS_DIR, 'male_scaler.joblib'))
            
            logger.info("Models and scalers loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def predict(self, voice_features, gender, age, bmi):
        """Make T2DM risk prediction based on voice features, gender, age and BMI"""
        try:
            if gender.lower() == 'female':
                # Prepare features for female model
                features = np.array([[
                    voice_features['meanF0'],
                    voice_features['stdevF0'],
                    voice_features['rapJitter'],
                    age,
                    bmi
                ]])
                # Scale features
                features_scaled = self.female_scaler.transform(features)
                voice_prob = self.female_model.predict_proba(features_scaled)[0][1]
                
                # Calculate BMI risk factor
                bmi_factor = self._calculate_bmi_risk(bmi)
                
                # Combine probabilities (as per paper: BMI only for women)
                final_prob = 0.75 * voice_prob + 0.25 * bmi_factor
                
            elif gender.lower() == 'male':
                # Prepare features for male model
                features = np.array([[
                    voice_features['meanInten'],
                    voice_features['apq11Shimmer'],
                    age,
                    bmi
                ]])
                # Scale features
                features_scaled = self.male_scaler.transform(features)
                voice_prob = self.male_model.predict_proba(features_scaled)[0][1]
                
                # Calculate risk factors
                age_factor = self._calculate_age_risk(age)
                bmi_factor = self._calculate_bmi_risk(bmi)
                
                # Combine probabilities (as per paper: both age and BMI for men)
                final_prob = 0.6 * voice_prob + 0.2 * age_factor + 0.2 * bmi_factor
            else:
                raise ValueError("Invalid gender specified")

            logger.info(f"Prediction details - Voice prob: {voice_prob:.3f}, Final prob: {final_prob:.3f}")
            return final_prob
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _calculate_age_risk(self, age):
        """Calculate risk factor based on age using sigmoid function"""
        base_age = 40
        scale = 0.1
        return 1 / (1 + np.exp(-scale * (age - base_age)))

    def _calculate_bmi_risk(self, bmi):
        """Calculate continuous risk factor based on BMI"""
        if bmi < 18.5:  # Underweight
            return 0.3
        elif bmi < 25:  # Normal
            return 0.2 + (bmi - 18.5) * 0.02
        elif bmi < 30:  # Overweight
            return 0.5 + (bmi - 25) * 0.06
        else:  # Obese
            return min(0.8 + (bmi - 30) * 0.01, 1.0)

# Initialize predictor
predictor = Predictor()

@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making T2DM risk predictions from voice recordings"""
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
        logger.info(f"Processing audio file: {audio_file.filename}")
        
        try:
            features = extract_audio_features(audio_file, form_data['gender'])
            if not features:
                return jsonify({'error': 'Failed to extract voice features'}), 400
            logger.info(f"Extracted features: {features}")
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return jsonify({'error': f'Failed to extract voice features: {str(e)}'}), 400
        
        # Make prediction
        try:
            risk_probability = predictor.predict(
                features, 
                form_data['gender'],
                age,
                bmi
            )
            logger.info(f"Prediction result: {risk_probability}")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

        return jsonify({
            'risk_probability': float(risk_probability),
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500