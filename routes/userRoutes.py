from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from models.user import UserSchema, add_user, get_device_profiles, verify_password
from bson import ObjectId
from db import mongo
from werkzeug.security import generate_password_hash

user_blueprint = Blueprint('user_routes', __name__)

@user_blueprint.route('/add-profile', methods=['POST'])
def create_user():
    user_data = request.json
    device_id = request.headers.get('X-Device-ID')
    
    if not device_id:
        return jsonify({"message": "Device ID is required"}), 400
        
    result = add_user(user_data, device_id)
    print(result)
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
        user['_id'] = str(user['_id'])
        user.pop('password', None)  # Remove password from response
        return jsonify({
            "message": "Login successful",
            "user": user
        }), 200
    
    return jsonify({"message": "Invalid email or password"}), 401

@user_blueprint.route('/device-profiles', methods=['GET'])
def get_recent_profiles():
    device_id = request.headers.get('X-Device-ID')
    
    if not device_id:
        return jsonify({"message": "Device ID is required"}), 400
    
    profiles = get_device_profiles(device_id)
    return jsonify(profiles), 200

@user_blueprint.route('/remove-device-profile/<user_id>', methods=['DELETE'])
def remove_device_profile(user_id):
    device_id = request.headers.get('X-Device-ID')
    
    if not device_id:
        return jsonify({"message": "Device ID is required"}), 400
        
    device_profiles = mongo.db.device_profiles
    result = device_profiles.delete_one({
        "device_id": device_id,
        "user_id": user_id
    })
    
    if result.deleted_count:
        return jsonify({"message": "Profile removed from device"}), 200
    return jsonify({"message": "Profile not found"}), 404

@user_blueprint.route('/change-password/<id>', methods=['PUT'])
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
    
    # Update password
    hashed_password = generate_password_hash(new_password)
    result = user_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"password": hashed_password}}
    )
    
    if result.modified_count:
        return jsonify({"message": "Password updated successfully"}), 200
    return jsonify({"message": "Failed to update password"}), 500

@user_blueprint.route('/get-all-profile', methods=['GET'])
def get_user():
    user_collection = mongo.db.users
    profiles = list(user_collection.find({}, {'password': 0}))  # Exclude password
    for profile in profiles:
        profile['_id'] = str(profile['_id'])
    return jsonify(profiles), 200

@user_blueprint.route('/profiles/<id>', methods=['PUT'])
def update_profile(id):
    schema = UserSchema(partial=True)
    user_collection = mongo.db.users
    
    try:
        updated_data = schema.load(request.json)
    except ValidationError as err:
        return {"errors": err.messages}, 400
    
    # If password is being updated, hash it
    if 'password' in updated_data:
        updated_data['password'] = generate_password_hash(updated_data['password'])
    
    result = user_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": updated_data}
    )
    
    if result.matched_count == 0:
        return {"message": "Profile not found"}, 404
    
    return {"message": "Profile updated successfully"}, 200

@user_blueprint.route('/profiles/<id>', methods=['DELETE'])
def delete_profile(id):
    user_collection = mongo.db.users
    result = user_collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        return {"message": "Profile not found"}, 404
    return {"message": "Profile deleted successfully"}, 200

@user_blueprint.route('/profiles/<id>/activate', methods=['PUT'])
def activate_profile(id):
    user_collection = mongo.db.users
    
    # Deactivate all profiles
    user_collection.update_many({}, {"$set": {"isActive": False}})
    
    # Activate the selected profile
    result = user_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"isActive": True}}
    )
    
    if result.matched_count == 0:
        return {"message": "Profile not found"}, 404
    
    return {"message": "Profile activated successfully", "_id": id}, 200