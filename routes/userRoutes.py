import os
from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from models.user import UserSchema, add_user, get_device_profiles, verify_password
from bson import ObjectId
from db import mongo
from werkzeug.security import generate_password_hash
from functools import wraps
import jwt
print(jwt.__file__)

import datetime

user_blueprint = Blueprint('user_routes', __name__)
SECRET_KEY = f"{os.getenv('SECRET_KEY')}" 

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        device_id = request.headers.get('X-Device-ID')
        
        if not device_id:
            return jsonify({'message': 'Device ID is required'}), 401
            
        if token:
            try:
                token = token.split(' ')[1]  
                data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                current_user = mongo.db.users.find_one({'_id': ObjectId(data['user_id'])})
                
                if not current_user:
                    return jsonify({'message': 'Invalid token'}), 401
                    
            except jwt.ExpiredSignatureError:
                return jsonify({'message': 'Token has expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'message': 'Invalid token'}), 401
        
        device_exists = mongo.db.device_profiles.find_one({'device_id': device_id})
        if not device_exists:
            return jsonify({'message': 'Device not registered'}), 401
            
        return f(*args, **kwargs)
    return decorated

@user_blueprint.route('/add-profile', methods=['POST'])
def create_user():
    user_data = request.json
    device_id = request.headers.get('X-Device-ID')
    if not device_id:
        return jsonify({"message": "Device ID is required"}), 400
        
    result = add_user(user_data, device_id)
    
    if isinstance(result, tuple) and result[1] == 201:
        token = jwt.encode({
            'user_id': str(result[0]['user_id']),
            'device_id': device_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }, SECRET_KEY, algorithm="HS256")

        
        response_data = {
            "message": "Profile created successfully",
            "token": token,
            "user": result[0]
        }
        return jsonify(response_data), 201
        
    return jsonify(result), result[1] if isinstance(result, tuple) else 200

@user_blueprint.route('/login-to-switch', methods=['POST'])
def login():
    data = request.json
    device_id = request.headers.get('X-Device-ID')
    
    if not device_id:
        return jsonify({"message": "Device ID is required"}), 400
    
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400
    
    user = verify_password(email, password, device_id)
    
    if user:
        token = jwt.encode({
            'user_id': str(user[0]['user_id']),
            'device_id': device_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }, SECRET_KEY, algorithm="HS256")

        
        user['_id'] = str(user['_id'])
        user.pop('password', None)
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": user
        }), 200
    
    return jsonify({"message": "Invalid email or password"}), 401

@user_blueprint.route('/device-profiles', methods=['GET'])
@auth_required
def get_recent_profiles():
    device_id = request.headers.get('X-Device-ID')
    profiles = get_device_profiles(device_id)
    return jsonify(profiles), 200

@user_blueprint.route('/remove-device-profile/<user_id>', methods=['DELETE'])
@auth_required
def remove_device_profile(user_id):
    device_id = request.headers.get('X-Device-ID')
    device_profiles = mongo.db.device_profiles
    result = device_profiles.delete_one({
        "device_id": device_id,
        "user_id": user_id
    })
    
    if result.deleted_count:
        return jsonify({"message": "Profile removed from device"}), 200
    return jsonify({"message": "Profile not found"}), 404

@user_blueprint.route('/change-password/<id>', methods=['PUT'])
@auth_required
def change_password(id):
    data = request.json
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not current_password or not new_password:
        return jsonify({"message": "Current and new passwords are required"}), 400
    
    user_collection = mongo.db.users
    user = user_collection.find_one({"_id": ObjectId(id)})
    
    if not user:
        return jsonify({"message": "User not found"}), 404
    
    if not verify_password(user["email"], current_password):
        return jsonify({"message": "Current password is incorrect"}), 401
    
    hashed_password = generate_password_hash(new_password)
    result = user_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"password": hashed_password}}
    )
    
    if result.modified_count:
        return jsonify({"message": "Password updated successfully"}), 200
    return jsonify({"message": "Failed to update password"}), 500

@user_blueprint.route('/get-profile/<profile_id>', methods=['GET'])
@auth_required
def get_profile(profile_id):
    """Get user profile by ID"""
    try:
        # Verify device has access to this profile
        device_id = request.headers.get('X-Device-ID')
        device_profile = mongo.db.device_profiles.find_one({
            'device_id': device_id,
            'user_id': profile_id
        })
        
        if not device_profile:
            return jsonify({
                'message': 'Profile not found or not associated with this device'
            }), 404

        # Get user data
        user = mongo.db.users.find_one({'_id': ObjectId(profile_id)})
        
        if not user:
            return jsonify({'message': 'Profile not found'}), 404
            
        # Remove sensitive data
        user.pop('password', None)
        user['_id'] = str(user['_id'])

        # Get latest health metrics
        latest_metrics = mongo.db.health_metrics.find_one(
            {'user_id': profile_id},
            sort=[('timestamp', -1)]
        )
        
        if latest_metrics:
            latest_metrics.pop('_id', None)
            user['latest_metrics'] = latest_metrics

        # Check if this is the active profile for the device
        is_active = device_profile.get('isActive', False)
        
        response_data = {
            'message': 'Profile retrieved successfully',
            'user': {
                **user,
                'isActive': is_active
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error fetching profile: {str(e)}")
        return jsonify({
            'message': 'Error retrieving profile',
            'error': str(e)
        }), 500
