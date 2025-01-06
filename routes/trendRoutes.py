from datetime import datetime, timedelta
import os
from flask import Blueprint, jsonify, request, current_app
from marshmallow import ValidationError
from functools import wraps
import jwt
from bson import ObjectId
from db import mongo
import numpy as np
from models.trendData import (
    TrendData, trend_schema, monthly_analysis_schema
)

trends = Blueprint('trend_routes', __name__)

def _calculate_trend(values):
    if len(values) < 2:
        return 'insufficient_data'
    
    slope = np.polyfit(range(len(values)), values, 1)[0]
    if abs(slope) < 0.001:
        return 'stable'
    return 'increasing' if slope > 0 else 'decreasing'

SECRET_KEY = f"{os.getenv('SECRET_KEY')}" 

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        print(request.headers, 'headers')
        token = request.headers.get('Authorization')
        device_id = request.headers.get('X-Device-ID')
        
        if not device_id:
            return jsonify({'message': 'Device ID is required'}), 401

        data = None
        if token:
            try:
                token = token.split(' ')[1]  
                print(token, 'token')
                data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                print(data)
                current_user = mongo.db.users.find_one({'_id': ObjectId(data['user_id'])})
                
                if not current_user:
                    return jsonify({'message': 'Invalid token'}), 401
                    
            except jwt.ExpiredSignatureError:
                return jsonify({'message': 'Token has expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'message': 'Invalid token'}), 401
        
        if not data:
            return jsonify({'message': 'Token is required'}), 401
        
        device_exists = mongo.db.device_profiles.find_one({'device_id': device_id})
        if not device_exists:
            return jsonify({'message': 'Device not registered'}), 401
            
        # Pass current_user_id as a keyword argument
        kwargs['current_user_id'] = data['user_id']
        
        return f(*args, **kwargs)
    return decorated

@trends.route('/trends/<user_id>', methods=['GET'])
def get_user_trends(user_id):
    try:
        days = int(request.args.get('days', 30))
        trends = list(TrendData.get_user_trends(user_id, days))

        if not trends:
            return jsonify({'message': 'No trend data available'}), 404

        # Convert timestamps to ISO format strings for response
        formatted_trends = []
        for trend in trends:
            trend_copy = trend.copy()
            if isinstance(trend_copy['timestamp'], datetime):
                trend_copy['timestamp'] = trend_copy['timestamp'].isoformat()
            formatted_trends.append(trend_copy)

        risk_values = [t['risk_level'] for t in trends]
        trend_data = {
            'trend_data': formatted_trends,
            'statistics': {
                'average_risk': float(np.mean(risk_values)),
                'std_dev': float(np.std(risk_values)),
                'trend_direction': _calculate_trend(risk_values),
                'min_risk': float(min(risk_values)),
                'max_risk': float(max(risk_values)),
                'total_recordings': len(trends)
            }
        }

        result = trend_schema.dump(trend_data)
        return jsonify(result)

    except ValidationError as err:
        return jsonify(err.messages), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

@trends.route('/trends/monthly-analysis/<user_id>', methods=['GET'])
# @auth_required
def get_monthly_analysis( user_id):
    try:
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        gender = user.get('gender', '').lower()
        trends = list(TrendData.get_user_trends(user_id, 30))

        if not trends:
            return jsonify({'message': 'No data available for analysis'}), 404

        features_to_analyze = ['meanF0', 'stdevF0', 'rapJitter'] if gender == 'female' else ['meanInten', 'apq11Shimmer']
        feature_stats = {
            feature: {
                'average': np.mean([t['features'].get(feature, 0) for t in trends]),
                'std_dev': np.std([t['features'].get(feature, 0) for t in trends]),
                'trend': _calculate_trend([t['features'].get(feature, 0) for t in trends])
            } for feature in features_to_analyze
        }

        analysis_data = {
            'monthly_analysis': {
                'feature_statistics': feature_stats,
                'recording_count': len(trends),
                'date_range': {
                    'start': trends[0]['timestamp'] if trends else None,
                    'end': trends[-1]['timestamp'] if trends else None
                }
            }
        }

        result = monthly_analysis_schema.dump(analysis_data)
        return jsonify(result)

    except ValidationError as err:
        return jsonify(err.messages), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500