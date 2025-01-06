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
                print(data)
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
            'exp': datetie.utcnow() + datetime.timedelta(days=1)
        }, SECRET_KEY, algorithm="HS256")

        
        response_data = {
            "message": "Profile created successfully",
            "token": token,
            "user": result[0]
        }
        return jsonify(response_data), 201
        
    return jsonify(result), result[1] if isinstance(result, tuple) else 200

@user_blueprint.route('/switch-profile', methods=['POST'])
@auth_required
def switch_profile():
    """Switch the active profile for a device"""
    try:
        # Get device ID from headers
        device_id = request.headers.get('X-Device-ID')
        if not device_id:
            return jsonify({
                "error": "X-Device-ID header is required"
            }), 400

        # Get user ID from request body
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({
                "error": "user_id is required in request body"
            }), 400
            
        user_id = data['user_id']
        
        try:
            user_id_obj = ObjectId(user_id)
        except:
            return jsonify({
                "error": "Invalid user_id format"
            }), 400

        device_profiles = mongo.db.device_profiles
        user_collection = mongo.db.users
            
        # First verify the user has a profile for this device
        profile = device_profiles.find_one({
            "device_id": device_id,
            "user_id": user_id
        })
        
        if not profile:
            return jsonify({
                "error": "Profile not found for this device"
            }), 404

        try:
            # Get all user IDs for this device
            device_user_profiles = device_profiles.find({"device_id": device_id})
            user_ids = [profile["user_id"] for profile in device_user_profiles]
            
            # Set isActive to False for all users of this device
            user_collection.update_many(
                {"_id": {"$in": [ObjectId(uid) for uid in user_ids]}},
                {"$set": {"isActive": False}}
            )
            
            # Set isActive to False for all profiles on this device
            device_profiles.update_many(
                {"device_id": device_id},
                {"$set": {"isActive": False}}
            )
            
            # Set the selected profile and user to active
            device_profiles.update_one(
                {
                    "device_id": device_id,
                    "user_id": user_id
                },
                {"$set": {"isActive": True}}
            )
            
            user_collection.update_one(
                {"_id": user_id_obj},
                {"$set": {"isActive": True}}
            )
            
            # Get updated user data
            updated_user = user_collection.find_one({"_id": user_id_obj})
            if not updated_user:
                raise Exception("Failed to fetch updated user data")

            return jsonify({
                "message": "Profile switched successfully",
                "user": {
                    "id": str(updated_user["_id"]),
                    "name": updated_user.get("name", ""),
                    "email": updated_user.get("email", ""),
                    "age": updated_user.get("age"),
                    "gender": updated_user.get("gender"),
                    "height": updated_user.get("height"),
                    "weight": updated_user.get("weight"),
                    "bmi": updated_user.get("bmi"),
                    "isActive": True
                }
            }), 200
            
        except Exception as e:
            print(f"Database error during profile update: {str(e)}")
            return jsonify({
                "error": "Failed to update profile status in database"
            }), 500
            
    except Exception as e:
        print(f"Server error in switch_profile: {str(e)}")
        return jsonify({
            "error": str(e) if str(e) else "An unexpected error occurred"
        }), 500
    
@user_blueprint.route('/device-profiles', methods=['GET'])
@auth_required
def get_recent_profiles():
    """
    Endpoint to get recent device profiles.
    
    Query Parameters:
        limit (int): Optional. Number of profiles to return (default: 5)
    
    Headers Required:
        X-Device-ID: Device identifier
    """
    try:
        # Get and validate device_id
        device_id = request.headers.get('X-Device-ID')
        if not device_id:
            return jsonify({
                "error": "X-Device-ID header is required"
            }), 400
            
            
        # Get profiles
        profiles = get_device_profiles(device_id)
        
        print(profiles)

        return jsonify({
            "success": True,
            "data": {
                "profiles": profiles,
                "total": len(profiles)
            }
        }), 200
        
    except ValueError as e:
        return jsonify({
            "error": str(e)
        }), 400
        
    except Exception as e:
        print(f"Unexpected error in get_recent_profiles: {str(e)}")  # Added logging
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500
    
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
