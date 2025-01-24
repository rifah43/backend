from flask import Blueprint, request, jsonify
import joblib
from marshmallow import ValidationError
import numpy as np
import os
import pandas as pd
from datetime import datetime
from db import mongo
from models.trendData import TrendDataSchema, VoiceFeaturesSchema
from utils.audioProcessing import extract_audio_features
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

predict_bp = Blueprint('predict', __name__)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        try:
            self.female_model = joblib.load(os.path.join(Config.MODELS_DIR, 'female_model.joblib'))
            self.male_model = joblib.load(os.path.join(Config.MODELS_DIR, 'male_model.joblib'))
            self.female_scaler = joblib.load(os.path.join(Config.MODELS_DIR, 'female_scaler.joblib'))
            self.male_scaler = joblib.load(os.path.join(Config.MODELS_DIR, 'male_scaler.joblib'))
            logger.info("Models and scalers loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def _engineer_features(self, voice_features, age, bmi):
        """Create engineered features matching the training data structure"""
        features = {
            'meanF0': voice_features['meanF0'],
            'stdevF0': voice_features['stdevF0'],
            'meanInten': voice_features['meanInten'],
            'stdevInten': voice_features['stdevInten'],
            'HNR': voice_features['hnr'],
            'localShimmer': voice_features['localShimmer'],
            'localDbShimmer': voice_features.get('localDbShimmer', 0),
            'apq3Shimmer': voice_features.get('apq3Shimmer', 0),
            'apq5Shimmer': voice_features.get('apq5Shimmer', 0),
            'apq11Shimmer': voice_features['apq11Shimmer'],
            'localJitter': voice_features['localJitter'],
            'rapJitter': voice_features['rapJitter'],
            'ppq5Jitter': voice_features.get('ppq5Jitter', 0),
            'pitch_intensity_ratio': voice_features['meanF0'] / voice_features['meanInten'],
            'jitter_shimmer_ratio': voice_features['rapJitter'] / voice_features['localShimmer'],
            'voice_irregularity': voice_features['stdevF0'] * voice_features['stdevInten'],
            'mean_shimmer': (voice_features['localShimmer'] + voice_features['apq11Shimmer']) / 2,
            'mean_jitter': (voice_features['localJitter'] + voice_features['rapJitter']) / 2,
            'Age_risk': self._calculate_age_risk(age),
            'BMI_risk': self._calculate_bmi_risk(bmi)
        }
        return np.array([[v for v in features.values()]])

    def predict(self, voice_features, gender, age, bmi):
        try:
            features = self._engineer_features(voice_features, age, bmi)
            
            if gender.lower() == 'female':
                features_scaled = self.female_scaler.transform(features)
                voice_prob = self.female_model.predict_proba(features_scaled)[0][1]
                bmi_factor = self._calculate_bmi_risk(bmi)
                final_prob = 0.75 * voice_prob + 0.25 * bmi_factor
                
            elif gender.lower() == 'male':
                features_scaled = self.male_scaler.transform(features)
                voice_prob = self.male_model.predict_proba(features_scaled)[0][1]
                age_factor = self._calculate_age_risk(age)
                bmi_factor = self._calculate_bmi_risk(bmi)
                final_prob = 0.6 * voice_prob + 0.2 * age_factor + 0.2 * bmi_factor
            else:
                raise ValueError("Invalid gender specified")

            logger.info(f"Prediction details - Voice prob: {voice_prob:.3f}, Final prob: {final_prob:.3f}")
            return final_prob
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _calculate_age_risk(self, age):
        base_age = 40
        scale = 0.1
        return 1 / (1 + np.exp(-scale * (age - base_age)))

    def _calculate_bmi_risk(self, bmi):
        if bmi < 18.5:
            return 0.3
        elif bmi < 25:
            return 0.2 + (bmi - 18.5) * 0.02
        elif bmi < 30:
            return 0.5 + (bmi - 25) * 0.06
        else:
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
         
        current_time = datetime.utcnow().replace(microsecond=0)
        timestamp_str = current_time.isoformat()
        
        # First validate features
        try:
            feature_schema = VoiceFeaturesSchema()
            validated_features = feature_schema.load(features)
            logger.info(f"Features validated successfully: {validated_features}")
        except ValidationError as e:
            logger.error(f"Feature validation error: {e.messages}")
            return jsonify({'error': f'Invalid feature data: {e.messages}'}), 400

        # Create trend data with validated features
        trend_data = {
            'user_id': form_data['user_id'],
            'risk_level': float(risk_probability),
            'features': validated_features,
            'timestamp': timestamp_str  # Use string timestamp
        }

        # Validate complete trend data
        try:
            trend_schema = TrendDataSchema()
            validated_trend = trend_schema.load(trend_data)
            logger.info(f"Trend data validated successfully: {validated_trend}")
        except ValidationError as e:
            logger.error(f"Trend data validation error: {e.messages}")
            return jsonify({'error': f'Invalid trend data: {e.messages}'}), 400

        # Save to database
        try:
            result = mongo.db.trend_data.insert_one(validated_trend)
            if not result.inserted_id:
                raise Exception("Database insertion failed")
            logger.info(f"Trend data saved with ID: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return jsonify({'error': 'Failed to save trend data'}), 500

        # Return success response
        return jsonify({
            'risk_probability': float(risk_probability),
            'features': validated_features,
            'timestamp': timestamp_str
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500
    