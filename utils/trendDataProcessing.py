# app/services/trend_analyzer.py
from datetime import datetime, timedelta
import uuid
from marshmallow import ValidationError
import numpy as np
from typing import Dict, List, Optional
from enum import Enum

from models.trendData import MonthlyTrendSchema, PredictionSchema

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TrendAnalyzer:
    def __init__(self, db):
        self.db = db
        self.prediction_schema = PredictionSchema()
        self.monthly_trend_schema = MonthlyTrendSchema()

    def analyze_prediction(self, features: Dict, risk_probability: float, gender: str) -> Dict:
        """Analyze a new prediction and update trends"""
        try:
            # Validate prediction data
            prediction_data = {
                "prediction_id": str(uuid.uuid4()),
                "features": features,
                "risk_probability": risk_probability,
                "gender": gender,
                "timestamp": datetime.utcnow()
            }
            validated_data = self.prediction_schema.load(prediction_data)

            # Store prediction and update trends
            with self.db.session.begin():
                # Store prediction
                self.db.predictions.insert_one(validated_data)
                
                # Update monthly trends
                self._update_monthly_trends(validated_data)

            # Return analysis results
            return self._get_immediate_analysis(validated_data)

        except ValidationError as err:
            raise ValueError(f"Invalid prediction data: {err.messages}")
        except Exception as e:
            raise Exception(f"Error analyzing prediction: {str(e)}")

    def _update_monthly_trends(self, prediction: Dict) -> None:
        """Update monthly trend analysis"""
        current_month = prediction["timestamp"].strftime('%Y-%m')
        
        # Get existing trends for the month
        existing_trend = self.db.monthly_trends.find_one({
            "user_id": prediction["user_id"],
            "month": current_month
        })

        if existing_trend:
            self._update_existing_trend(existing_trend, prediction)
        else:
            self._create_new_trend(prediction)

    def _update_existing_trend(self, existing_trend: Dict, new_prediction: Dict) -> None:
        """Update existing monthly trend incrementally"""
        n = existing_trend["statistics"]["recording_count"]
        features = new_prediction["features"]
        
        updates = {
            "statistics.recording_count": n + 1,
            "statistics.average_risk": self._update_running_average(
                existing_trend["statistics"]["average_risk"],
                new_prediction["risk_probability"],
                n
            ),
            "updated_at": datetime.utcnow()
        }

        # Update feature analysis
        for feature_name, feature_value in features.items():
            current_analysis = existing_trend["feature_analysis"].get(feature_name, {})
            
            updates[f"feature_analysis.{feature_name}"] = {
                "average": self._update_running_average(
                    current_analysis.get("average", feature_value),
                    feature_value,
                    n
                ),
                "min_value": min(current_analysis.get("min_value", float('inf')), feature_value),
                "max_value": max(current_analysis.get("max_value", float('-inf')), feature_value),
                "trend_direction": self._determine_trend([
                    current_analysis.get("last_value", feature_value),
                    feature_value
                ]),
                "last_value": feature_value,
                "last_updated": datetime.utcnow()
            }

        self.db.monthly_trends.update_one(
            {"_id": existing_trend["_id"]},
            {"$set": updates}
        )

    def _create_new_trend(self, prediction: Dict) -> None:
        """Create new monthly trend entry"""
        features = prediction["features"]
        now = datetime.utcnow()
        
        trend_data = {
            "trend_id": str(uuid.uuid4()),
            "user_id": prediction["user_id"],
            "month": prediction["timestamp"].strftime('%Y-%m'),
            "statistics": {
                "recording_count": 1,
                "average_risk": prediction["risk_probability"]
            },
            "feature_analysis": {
                name: {
                    "average": value,
                    "min_value": value,
                    "max_value": value,
                    "trend_direction": "insufficient_data",
                    "last_value": value,
                    "last_updated": now
                }
                for name, value in features.items()
            },
            "gender_specific_metrics": self._calculate_gender_metrics(
                features,
                prediction["gender"]
            ),
            "created_at": now,
            "updated_at": now
        }

        try:
            validated_trend = self.monthly_trend_schema.load(trend_data)
            self.db.monthly_trends.insert_one(validated_trend)
        except ValidationError as err:
            raise ValueError(f"Invalid trend data: {err.messages}")

    def get_trend_analysis(self, user_id: str, timeframe: str = 'monthly') -> Dict:
        """Get trend analysis for specified timeframe"""
        end_date = datetime.utcnow()
        
        if timeframe == 'daily':
            start_date = end_date - timedelta(days=1)
        elif timeframe == 'weekly':
            start_date = end_date - timedelta(days=7)
        else:
            start_date = end_date - timedelta(days=30)

        predictions = list(self.db.predictions.find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }).sort("timestamp", 1))

        if not predictions:
            return None

        return {
            "risk_analysis": self._analyze_risk_trends(predictions),
            "feature_analysis": self._analyze_feature_trends(predictions),
            "compliance": self._analyze_compliance(predictions, timeframe),
            "time_series": self._prepare_time_series_data(predictions)
        }

    def _analyze_risk_trends(self, predictions: List[Dict]) -> Dict:
        """Analyze risk probability trends"""
        risk_values = [p["risk_probability"] for p in predictions]
        
        return {
            "current_risk": risk_values[-1] if risk_values else None,
            "average_risk": np.mean(risk_values) if risk_values else None,
            "risk_distribution": self._calculate_risk_distribution(risk_values),
            "trend_direction": self._determine_trend(risk_values),
            "peak_risk_times": self._identify_peak_risk_times(predictions)
        }

    def _analyze_feature_trends(self, predictions: List[Dict]) -> Dict:
        """Analyze acoustic feature trends"""
        feature_trends = {}
        
        if not predictions:
            return feature_trends

        for feature in predictions[0]["features"].keys():
            values = [p["features"][feature] for p in predictions]
            feature_trends[feature] = {
                "current_value": values[-1],
                "average": np.mean(values),
                "std_dev": np.std(values),
                "trend_direction": self._determine_trend(values),
                "range": [min(values), max(values)]
            }

        return feature_trends

    @staticmethod
    def _determine_trend(values: List[float], min_points: int = 2) -> str:
        """Determine trend direction from values"""
        if len(values) < min_points:
            return "insufficient_data"
            
        first_value = values[0]
        last_value = values[-1]
        
        if last_value > first_value * 1.05:  # 5% threshold
            return "increasing"
        elif last_value < first_value * 0.95:
            return "decreasing"
        return "stable"

    def _prepare_time_series_data(self, predictions: List[Dict]) -> Dict:
        """Prepare data for time series visualization"""
        return {
            "timestamps": [p["timestamp"] for p in predictions],
            "risk_values": [p["risk_probability"] for p in predictions],
            "feature_values": {
                feature: [p["features"][feature] for p in predictions]
                for feature in predictions[0]["features"].keys()
            } if predictions else {}
        }

    @staticmethod
    def _update_running_average(old_avg: float, new_value: float, n: int) -> float:
        """Update running average with new value"""
        return (old_avg * n + new_value) / (n + 1)

    def _calculate_gender_metrics(self, features: Dict, gender: str) -> Dict:
        """Calculate gender-specific voice metrics"""
        if gender == "female":
            pitch_range = (100, 300)  # Female pitch range
        else:
            pitch_range = (75, 200)   # Male pitch range

        return {
            "pitch_range_compliance": 1 if pitch_range[0] <= features["meanF0"] <= pitch_range[1] else 0,
            "intensity_stability": 1 - (features["stdevIntensity"] / 100),
            "voice_quality": (
                0.3 * features["hnr"] +
                0.2 * (1 - features["localJitter"]) +
                0.2 * (1 - features["localShimmer"]) +
                0.3 * (1 - features["meanVTI"])
            )
        }

    def _calculate_risk_distribution(self, risk_values: List[float]) -> Dict:
        """Calculate risk level distribution"""
        if not risk_values:
            return {level.value: 0 for level in RiskLevel}
            
        distribution = {level.value: 0 for level in RiskLevel}
        
        for value in risk_values:
            if value >= 0.7:
                distribution[RiskLevel.HIGH.value] += 1
            elif value >= 0.3:
                distribution[RiskLevel.MEDIUM.value] += 1
            else:
                distribution[RiskLevel.LOW.value] += 1
                
        total = len(risk_values)
        return {level: (count / total) * 100 for level, count in distribution.items()}

    def _identify_peak_risk_times(self, predictions: List[Dict]) -> Dict:
        """Identify times when risk is typically highest"""
        time_risks = {}
        
        for pred in predictions:
            hour = pred["timestamp"].hour
            if hour not in time_risks:
                time_risks[hour] = []
            time_risks[hour].append(pred["risk_probability"])

        avg_risks = {
            hour: np.mean(risks)
            for hour, risks in time_risks.items()
        }

        if not avg_risks:
            return None

        peak_hour = max(avg_risks.items(), key=lambda x: x[1])[0]
        return {
            "peak_hour": peak_hour,
            "hourly_risks": avg_risks
        }