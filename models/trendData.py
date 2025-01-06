from datetime import datetime, timedelta
from db import mongo
from bson import ObjectId
from marshmallow import Schema, fields, post_load, validate, ValidationError

class VoiceFeaturesSchema(Schema):
    meanF0 = fields.Float(required=True, validate=validate.Range(min=0))
    stdevF0 = fields.Float(required=True, validate=validate.Range(min=0))
    meanInten = fields.Float(required=True, validate=validate.Range(min=0))
    rapJitter = fields.Float(required=True, validate=validate.Range(min=0))
    apq11Shimmer = fields.Float(required=True, validate=validate.Range(min=0))
    stdevInten = fields.Float(required=True, validate=validate.Range(min=0))
    hnr = fields.Float(required=True, validate=validate.Range(min=0))
    localJitter = fields.Float(required=True, validate=validate.Range(min=0))
    localShimmer = fields.Float(required=True, validate=validate.Range(min=0))

    @post_load
    def make_features(self, data, **kwargs):
        return data

class TrendDataSchema(Schema):
    user_id = fields.String(required=True)
    # Change timestamp field to use string format for validation
    timestamp = fields.String(required=True)
    risk_level = fields.Float(required=True, validate=validate.Range(min=0, max=1))
    features = fields.Nested(VoiceFeaturesSchema, required=True)

    @post_load
    def convert_timestamp(self, data, **kwargs):
        # Convert the ISO string timestamp to datetime object after validation
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            try:
                data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except ValueError as e:
                raise ValidationError({'timestamp': ['Invalid datetime format']})
        return data

class TrendStatisticsSchema(Schema):
    average_risk = fields.Float(required=True)
    std_dev = fields.Float(required=True)
    trend_direction = fields.String(required=True, 
        validate=validate.OneOf(['increasing', 'decreasing', 'stable', 'insufficient_data']))
    min_risk = fields.Float(required=True)
    max_risk = fields.Float(required=True)
    total_recordings = fields.Integer(required=True)

class TrendResponseSchema(Schema):
    trend_data = fields.List(fields.Nested(TrendDataSchema), required=True)
    statistics = fields.Nested(TrendStatisticsSchema, required=True)

class FeatureStatsSchema(Schema):
    average = fields.Float(required=True)
    std_dev = fields.Float(required=True)
    trend = fields.String(required=True, 
        validate=validate.OneOf(['increasing', 'decreasing', 'stable', 'insufficient_data']))

class MonthlyAnalysisSchema(Schema):
    feature_statistics = fields.Dict(keys=fields.String(), values=fields.Nested(FeatureStatsSchema))
    recording_count = fields.Integer(required=True)
    date_range = fields.Dict(keys=fields.String(), values=fields.DateTime())

trend_schema = TrendResponseSchema()
monthly_analysis_schema = MonthlyAnalysisSchema()

class TrendData:
    def __init__(self, user_id, risk_level, recording_id, features):
        self.user_id = str(user_id)  # Ensure user_id is string
        self.risk_level = float(risk_level)  # Ensure risk_level is float
        self.features = features
        self.timestamp = datetime.utcnow()

    def save(self):
        trend_data = {
            'user_id': self.user_id,
            'risk_level': self.risk_level,
            'features': self.features,
            'timestamp': self.timestamp
        }
        return mongo.db.trend_data.insert_one(trend_data)

    @staticmethod
    def get_user_trends(user_id, days=30):
        try:
            # Convert user_id to string if it isn't already
            user_id = str(user_id)
            
            # Calculate date range
            end_date = datetime.utcnow() 
            start_date = end_date - timedelta(days=days)
            
            # Create query
            query = {
                'user_id': user_id,
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            # Execute query with logging
            print(f"Executing query: {query}")
            cursor = mongo.db.trend_data.find(query).sort('timestamp', 1)
            
            # Convert cursor to list for easier handling
            trends = list(cursor)
            print(f"Retrieved {len(trends)} trend records for user {user_id}")
            
            return trends
            
        except Exception as e:
            print(f"Error retrieving trends: {str(e)}")
            return []

    @staticmethod
    def calculate_statistics(user_id):
        trends = TrendData.get_user_trends(user_id)
        if not trends:
            return {
                'average_risk': 0.0,
                'std_dev': 0.0,
                'min_risk': 0.0,
                'max_risk': 0.0,
                'trend_direction': 'insufficient_data',
                'total_recordings': 0
            }
            
        risk_levels = [t['risk_level'] for t in trends]
        
        # Calculate statistics
        avg_risk = sum(risk_levels) / len(risk_levels)
        std_dev = (sum((x - avg_risk) ** 2 for x in risk_levels) / len(risk_levels)) ** 0.5
        
        # Determine trend direction
        if len(risk_levels) < 2:
            trend = 'insufficient_data'
        else:
            first_half = sum(risk_levels[:len(risk_levels)//2]) / (len(risk_levels)//2)
            second_half = sum(risk_levels[len(risk_levels)//2:]) / (len(risk_levels) - len(risk_levels)//2)
            if abs(second_half - first_half) < 0.05:  # 5% threshold
                trend = 'stable'
            else:
                trend = 'increasing' if second_half > first_half else 'decreasing'
        
        return {
            'average_risk': avg_risk,
            'std_dev': std_dev,
            'min_risk': min(risk_levels),
            'max_risk': max(risk_levels),
            'trend_direction': trend,
            'total_recordings': len(trends)
        }