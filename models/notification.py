from datetime import datetime, timedelta
import random
from db import mongo
from marshmallow import Schema, fields, validate

class NotificationSchema(Schema):
    id = fields.String(dump_only=True)
    user_id = fields.String(required=True)
    type = fields.String(required=True, validate=validate.OneOf(['daily', 'monthly', 'health_tip']))
    title = fields.String(required=True)
    message = fields.String(required=True)
    scheduled_time = fields.DateTime(required=True)
    is_read = fields.Boolean(default=False)
    created_at = fields.DateTime(default=lambda: datetime.utcnow())

class NotificationSettingsSchema(Schema):
    user_id = fields.String(required=True)
    daily_reminder_time = fields.Time(required=True)
    notifications_enabled = fields.Boolean(default=True)
    health_tips_enabled = fields.Boolean(default=True)
    monthly_summary_enabled = fields.Boolean(default=True)

class NotificationService:
    @staticmethod
    def schedule_daily_reminder(user_id, reminder_time):
        notification = {
            'user_id': user_id,
            'type': 'daily',
            'title': 'Voice Analysis Reminder',
            'message': 'Time for your daily voice recording!',
            'scheduled_time': reminder_time,
            'is_read': False,
            'created_at': datetime.utcnow()
        }
        return mongo.db.notifications.insert_one(notification)

    @staticmethod
    def schedule_monthly_summary(user_id):
        today = datetime.now()
        next_month = today.replace(day=1) + timedelta(days=32)
        next_month_first = next_month.replace(day=1)
        
        notification = {
            'user_id': user_id,
            'type': 'monthly',
            'title': 'Monthly Health Summary',
            'message': 'Your health analysis for the past month is ready!',
            'scheduled_time': next_month_first,
            'is_read': False,
            'created_at': datetime.utcnow()
        }
        return mongo.db.notifications.insert_one(notification)

    @staticmethod
    def schedule_health_tip(user_id):
        health_tips = [
            "Regular exercise can help reduce T2DM risk",
            "Maintaining a balanced diet is key to preventing diabetes", 
            "Adequate sleep helps regulate blood sugar levels",
            "Stay hydrated to help maintain healthy blood sugar levels"
        ]
        
        notification = {
            'user_id': user_id,
            'type': 'health_tip',
            'title': 'Health Tip',
            'message': random.choice(health_tips),
            'scheduled_time': datetime.now() + timedelta(days=random.randint(1, 7)),
            'is_read': False,
            'created_at': datetime.utcnow()
        }
        return mongo.db.notifications.insert_one(notification)

    @staticmethod
    def get_notifications(user_id):
        return list(mongo.db.notifications.find({'user_id': user_id}).sort('created_at', -1))

    @staticmethod
    def mark_as_read(notification_id):
        return mongo.db.notifications.update_one(
            {'_id': notification_id},
            {'$set': {'is_read': True}}
        )

    @staticmethod
    def get_settings(user_id):
        return mongo.db.notification_settings.find_one({'user_id': user_id})

    @staticmethod
    def update_settings(settings):
        return mongo.db.notification_settings.update_one(
            {'user_id': settings['user_id']},
            {'$set': settings},
            upsert=True
        )