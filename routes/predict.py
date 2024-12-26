from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
from utils.audioProcessing import process_audio
from models.trendData import TrendData
from db import db

predict_bp = Blueprint('predict', __name__)

class Predictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self._load_latest_models()
    
    def _load_latest_models(self):
        with open(os.path.join(self.models_dir, 'version_info.json'), 'r') as f:
            version_info = json.load(f)
        
        version = version_info['version']
        self.female_model = joblib.load(
            os.path.join(self.models_dir, f'female_model_v{version}.joblib'))
        self.male_model = joblib.load(
            os.path.join(self.models_dir, f'male_model_v{version}.joblib'))
        self.version = version
    
    def predict(self, voice_features, gender):
        if gender == 'Female':
            features = np.array([[
                voice_features['meanF0'],
                voice_features['stdevF0'],
                voice_features['rapJitter']
            ]])
            model = self.female_model
        else:
            features = np.array([[
                voice_features['meanInten'],
                voice_features['apq11Shimmer']
            ]])
            model = self.male_model
        
        probability = model.predict_proba(features)[0][1]
        self._store_prediction(voice_features, gender, probability)
        return probability
    
    def _store_prediction(self, features, gender, probability):
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gender': gender,
            'prediction': probability,
            **features
        }
        df = pd.DataFrame([data])
        df.to_csv('dataset/collected_data.csv', mode='a', header=False, index=False)

predictor = Predictor()

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        user_id = request.form.get('user_id')
        gender = request.form.get('gender')
        
        # Process audio file to extract features
        features = process_audio(audio_file)
        
        # Make prediction
        risk_probability = predictor.predict(features, gender)
        
        # Store trend data
        # trend_data = TrendData(
        #     user_id=user_id,
        #     **features,
        #     risk_probability=risk_probability
        # )
        # db.session.add(trend_data)
        # db.session.commit()
        
        return jsonify({
            'risk_probability': float(risk_probability),
            'features': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500