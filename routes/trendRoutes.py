from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from marshmallow import ValidationError
from functools import wraps
import jwt
from utils.trendDataProcessing import TrendAnalyzer

trends = Blueprint('trend_routes', __name__)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Missing authorization token'}), 401
            
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(current_user_id, *args, **kwargs)
    return decorated

@trends.route('/analyze', methods=['GET'])
@token_required
def analyze_trends(current_user_id):
    """Get trend analysis for user"""
    timeframe = request.args.get('timeframe', 'monthly')
    if timeframe not in ['daily', 'weekly', 'monthly']:
        return jsonify({'error': 'Invalid timeframe. Must be daily, weekly, or monthly'}), 400
        
    analyzer = TrendAnalyzer(current_app.db)
    
    try:
        analysis = analyzer.get_trend_analysis(current_user_id, timeframe)
        if not analysis:
            return jsonify({'error': 'No trend data available'}), 404
            
        return jsonify(analysis), 200
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing trends: {str(e)}'}), 500

@trends.route('/prediction', methods=['POST'])
@token_required
def create_prediction(current_user_id):
    """Create new voice prediction and update trends"""
    try:
        data = request.get_json()
        
        if not all(k in data for k in ['features', 'risk_probability', 'gender']):
            return jsonify({'error': 'Missing required fields'}), 400
            
        analyzer = TrendAnalyzer(current_app.db)
        result = analyzer.analyze_prediction(
            features=data['features'],
            risk_probability=data['risk_probability'],
            gender=data['gender']
        )
        
        return jsonify(result), 201
        
    except ValidationError as err:
        return jsonify({'error': 'Validation error', 'details': err.messages}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing prediction: {str(e)}'}), 500

@trends.route('/monthly/<month>', methods=['GET'])
@token_required
def get_monthly_trend(current_user_id, month):
    """Get monthly trend summary"""
    try:
        month_format = '%Y-%m'
        # Validate month format
        try:
            datetime.strptime(month, month_format)
        except ValueError:
            return jsonify({'error': 'Invalid month format. Use YYYY-MM'}), 400
            
        monthly_trend = current_app.db.monthly_trends.find_one({
            "user_id": current_user_id,
            "month": month
        })
        
        if not monthly_trend:
            return jsonify({'error': 'No trend data available for specified month'}), 404
            
        # Remove MongoDB _id field
        monthly_trend.pop('_id', None)
        return jsonify(monthly_trend), 200
        
    except Exception as e:
        return jsonify({'error': f'Error fetching monthly trend: {str(e)}'}), 500

@trends.route('/compliance', methods=['GET'])
@token_required
def get_compliance_stats(current_user_id):
    """Get recording compliance statistics"""
    timeframe = request.args.get('timeframe', 'monthly')
    if timeframe not in ['daily', 'weekly', 'monthly']:
        return jsonify({'error': 'Invalid timeframe'}), 400
        
    try:
        analyzer = TrendAnalyzer(current_app.db)
        analysis = analyzer.get_trend_analysis(current_user_id, timeframe)
        
        if not analysis:
            return jsonify({'error': 'No compliance data available'}), 404
            
        return jsonify(analysis['compliance']), 200
        
    except Exception as e:
        return jsonify({'error': f'Error fetching compliance stats: {str(e)}'}), 500

@trends.route('/time-series', methods=['GET'])
@token_required
def get_time_series_data(current_user_id):
    """Get time series data for visualization"""
    timeframe = request.args.get('timeframe', 'monthly')
    if timeframe not in ['daily', 'weekly', 'monthly']:
        return jsonify({'error': 'Invalid timeframe'}), 400
        
    try:
        analyzer = TrendAnalyzer(current_app.db)
        analysis = analyzer.get_trend_analysis(current_user_id, timeframe)
        
        if not analysis:
            return jsonify({'error': 'No time series data available'}), 404
            
        return jsonify(analysis['time_series']), 200
        
    except Exception as e:
        return jsonify({'error': f'Error fetching time series data: {str(e)}'}), 500

@trends.errorhandler(Exception)
def handle_error(error):
    """Global error handler for trend routes"""
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500