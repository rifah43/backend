from datetime import datetime, timedelta
import random
from typing import List, Optional
from bson import ObjectId
from db import mongo
from marshmallow import Schema, fields, validate, ValidationError

class NotificationSchema(Schema):
    id = fields.String(dump_only=True)
    user_id = fields.String(required=True)
    type = fields.String(required=True, validate=validate.OneOf(['daily', 'monthly', 'health_tip']))
    title = fields.String(required=True, validate=validate.Length(min=1, max=100))
    message = fields.String(required=True, validate=validate.Length(min=1, max=500))
    scheduled_time = fields.DateTime(required=True)
    is_read = fields.Boolean(default=False)
    created_at = fields.DateTime(default=lambda: datetime.utcnow())

class NotificationSettingsSchema(Schema):
    user_id = fields.String(required=True)
    daily_reminder_time = fields.Time(required=True)
    notifications_enabled = fields.Boolean(default=True)
    health_tips_enabled = fields.Boolean(default=True)
    monthly_summary_enabled = fields.Boolean(default=True)
    updated_at = fields.DateTime(dump_only=True, default=lambda: datetime.utcnow())

class NotificationService:
    HEALTH_TIPS = [
        "Regular exercise can help reduce T2DM risk",
        "Maintaining a balanced diet is key to preventing diabetes",
        "Adequate sleep helps regulate blood sugar levels",
        "Stay hydrated to help maintain healthy blood sugar levels",
        "Regular voice recordings help monitor your health progress"
    ]

    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        return bool(mongo.db.users.find_one({'_id': ObjectId(user_id)}))

    @classmethod
    def schedule_daily_reminder(cls, user_id: str, reminder_time: datetime) -> ObjectId:
        if not cls.validate_user_id(user_id):
            raise ValueError(f"Invalid user_id: {user_id}")

        notification = {
            'user_id': user_id,
            'type': 'daily',
            'title': 'Voice Analysis Reminder',
            'message': 'Time for your daily voice recording!',
            'scheduled_time': reminder_time,
            'is_read': False,
            'created_at': datetime.utcnow()
        }
        
        result = mongo.db.notifications.insert_one(notification)
        return result.inserted_id

    @classmethod
    def schedule_monthly_summary(cls, user_id: str) -> ObjectId:
        if not cls.validate_user_id(user_id):
            raise ValueError(f"Invalid user_id: {user_id}")

        today = datetime.now()
        next_month = today.replace(day=1) + timedelta(days=32)
        next_month_first = next_month.replace(day=1, hour=9, minute=0, second=0, microsecond=0)
        
        notification = {
            'user_id': user_id,
            'type': 'monthly',
            'title': 'Monthly Health Summary',
            'message': 'Your health analysis for the past month is ready!',
            'scheduled_time': next_month_first,
            'is_read': False,
            'created_at': datetime.utcnow()
        }
        
        result = mongo.db.notifications.insert_one(notification)
        return result.inserted_id

    @classmethod
    def schedule_health_tip(cls, user_id: str) -> ObjectId:
        if not cls.validate_user_id(user_id):
            raise ValueError(f"Invalid user_id: {user_id}")

        notification = {
            'user_id': user_id,
            'type': 'health_tip',
            'title': 'Health Tip',
            'message': random.choice(cls.HEALTH_TIPS),
            'scheduled_time': datetime.now() + timedelta(days=random.randint(1, 7)),
            'is_read': False,
            'created_at': datetime.utcnow()
        }
        
        result = mongo.db.notifications.insert_one(notification)
        return result.inserted_id

    @staticmethod
    def get_notifications(user_id: str) -> List[dict]:
        return list(mongo.db.notifications.find(
            {'user_id': user_id},
            {'_id': 1, 'type': 1, 'title': 1, 'message': 1, 'scheduled_time': 1, 'is_read': 1}
        ).sort('created_at', -1))

    @staticmethod
    def mark_as_read(notification_id: str) -> bool:
        result = mongo.db.notifications.update_one(
            {'_id': ObjectId(notification_id)},
            {'$set': {'is_read': True}}
        )
        return result.modified_count > 0

    @staticmethod
    def get_settings(user_id: str) -> Optional[dict]:
        return mongo.db.notification_settings.find_one({'user_id': user_id})

    @staticmethod
    def update_settings(settings: dict) -> bool:
        settings['updated_at'] = datetime.utcnow()
        result = mongo.db.notification_settings.update_one(
            {'user_id': settings['user_id']},
            {'$set': settings},
            upsert=True
        )
        return result.modified_count > 0 or result.upserted_id is not None