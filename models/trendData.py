# app/models/schemas.py
from marshmallow import Schema, fields, validate, ValidationError
from datetime import datetime

class VoiceFeaturesSchema(Schema):
    """Voice acoustic features schema"""
    meanF0 = fields.Float(required=True, validate=validate.Range(min=0))
    stdevF0 = fields.Float(required=True, validate=validate.Range(min=0))
    meanIntensity = fields.Float(required=True, validate=validate.Range(min=0))
    stdevIntensity = fields.Float(required=True, validate=validate.Range(min=0))
    hnr = fields.Float(required=True)
    localJitter = fields.Float(required=True, validate=validate.Range(min=0))
    rapJitter = fields.Float(required=True, validate=validate.Range(min=0))
    localShimmer = fields.Float(required=True, validate=validate.Range(min=0))
    apq11Shimmer = fields.Float(required=True, validate=validate.Range(min=0))
    phonationTime = fields.Float(required=True, validate=validate.Range(min=0))
    meanVTI = fields.Float(required=True)

class PredictionSchema(Schema):
    """Voice analysis prediction schema"""
    prediction_id = fields.Str(required=True)
    user_id = fields.Str(required=True)
    features = fields.Nested(VoiceFeaturesSchema, required=True)
    risk_probability = fields.Float(required=True, validate=validate.Range(min=0, max=1))
    gender = fields.Str(required=True, validate=validate.OneOf(['male', 'female']))
    timestamp = fields.DateTime(required=True)
    audio_file_path = fields.Str(required=True)

class TrendStatisticsSchema(Schema):
    """Trend statistics schema"""
    average = fields.Float(required=True)
    min_value = fields.Float(required=True)
    max_value = fields.Float(required=True)
    std_dev = fields.Float(required=True)
    trend_direction = fields.Str(required=True, 
                               validate=validate.OneOf(['increasing', 'decreasing', 'stable', 'insufficient_data']))
    last_updated = fields.DateTime(required=True)

class MonthlyTrendSchema(Schema):
    """Monthly trend analysis schema"""
    trend_id = fields.Str(required=True)
    user_id = fields.Str(required=True)
    month = fields.Str(required=True)
    statistics = fields.Dict(keys=fields.Str(), values=fields.Float(), required=True)
    feature_analysis = fields.Dict(keys=fields.Str(), values=fields.Nested(TrendStatisticsSchema), required=True)
    gender_specific_metrics = fields.Dict(keys=fields.Str(), values=fields.Float(), required=True)
    timestamp = fields.DateTime(required=True)

class ComplianceSchema(Schema):
    """Recording compliance schema"""
    user_id = fields.Str(required=True)
    period = fields.Str(required=True, validate=validate.OneOf(['daily', 'weekly', 'monthly']))
    target_recordings = fields.Int(required=True)
    actual_recordings = fields.Int(required=True)
    compliance_rate = fields.Float(required=True, validate=validate.Range(min=0, max=100))
    streak = fields.Int(required=True)
    last_recording = fields.DateTime(required=True)