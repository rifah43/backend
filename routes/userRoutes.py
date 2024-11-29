from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from models.user import UserSchema, add_user
from bson import ObjectId
from db import mongo

user_blueprint = Blueprint('user_routes', __name__)


@user_blueprint.route('/add-profile', methods=['POST'])
def create_user():
    user_data = request.json
    result = add_user(user_data)
    return jsonify(result), result[1] if isinstance(result, tuple) else 200


@user_blueprint.route('/get-all-profile', methods=['GET'])
def get_user():
    user_collection = mongo.db.users
    profiles = list(user_collection.find({}))  
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

    result = user_collection.update_one({"_id": ObjectId(id)}, {"$set": updated_data})
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

    user_collection.update_many({}, {"$set": {"isActive": False}})

    result = user_collection.update_one({"_id": ObjectId(id)}, {"$set": {"isActive": True}})
    
    if result.matched_count == 0:
        return {"message": "Profile not found"}, 404

    return {"message": "Profile activated successfully", "_id": id}, 200