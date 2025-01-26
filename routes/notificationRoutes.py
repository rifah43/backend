from flask import Blueprint, request, jsonify
from models.notification import NotificationSchema, NotificationSettingsSchema
from datetime import datetime, time
from db import mongo

notification_routes = Blueprint('notification', __name__)

@notification_routes.route('/settings', methods=['GET', 'POST'])
def notification_settings():
    if request.method == 'POST':
        settings = NotificationSettingsSchema().load(request.json)
        mongo.db.notification_settings.update_one(
            {'user_id': settings['user_id']},
            {'$set': settings},
            upsert=True
        )
        return jsonify({'message': 'Settings updated successfully'})
    
    user_id = request.args.get('user_id')
    settings = mongo.db.notification_settings.find_one({'user_id': user_id})
    return jsonify(NotificationSettingsSchema().dump(settings))

@notification_routes.route('/notifications', methods=['GET'])
def get_notifications():
    user_id = request.args.get('user_id')
    notifications = list(mongo.db.notifications.find({'user_id': user_id}))
    return jsonify(NotificationSchema(many=True).dump(notifications))

@notification_routes.route('/notifications/mark-read/<notification_id>', methods=['POST'])
def mark_notification_read(notification_id):
    mongo.db.notifications.update_one(
        {'_id': notification_id},
        {'$set': {'is_read': True}}
    )
    return jsonify({'message': 'Notification marked as read'})