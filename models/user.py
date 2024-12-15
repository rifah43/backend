from bson import ObjectId
from db import mongo
from marshmallow import Schema, fields, validate, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

class UserSchema(Schema):
    name = fields.Str(required=True)
    age = fields.Int(required=True, validate=validate.Range(min=0))
    gender = fields.Str(required=True, validate=validate.OneOf(['Male','Female','Other']))
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=6))
    height = fields.Float(required=True)
    weight = fields.Float(required=True)
    bmi = fields.Float(required=True)
    isActive = fields.Boolean(missing=False)

class DeviceProfileSchema(Schema):
    device_id = fields.Str(required=True)
    user_id = fields.Str(required=True)

def add_user(user_data, device_id=None):
    schema = UserSchema()

    try:
        validated_data = schema.load(user_data)
    except ValidationError as err:
        return {"errors": err.messages}, 400

    try:
        user_collection = mongo.db.get_collection('users')
        device_profiles = mongo.db.get_collection('device_profiles')
    except Exception as e:
        raise ValueError(f"Error accessing the collection: {e}")

    # Check for existing email
    existing_user = user_collection.find_one({"email": validated_data["email"]})
    if existing_user:
        return {"message": "User with this email already exists."}, 400

    # Hash the password before storing
    validated_data["password"] = generate_password_hash(validated_data["password"])
    
    user_id = user_collection.insert_one(validated_data).inserted_id

    # If device_id is provided, add to device profiles
    if device_id:
        device_profiles.insert_one({
            "device_id": device_id,
            "user_id": str(user_id),
        })

    return {"message": "User created", "user_id": str(user_id)}, 201

def verify_password(email, password, device_id=None):
    user_collection = mongo.db.users
    device_profiles = mongo.db.device_profiles
    
    user = user_collection.find_one({"email": email})
    
    if user and check_password_hash(user["password"], password):
        user_collection.update_one(
            {"_id": user["_id"]},
        )

        # Update device profiles if device_id is provided
        if device_id:
            device_profiles.update_one(
                {
                    "device_id": device_id,
                    "user_id": str(user["_id"])
                },
                upsert=True
            )
        
        return user
    return None

def get_device_profiles(device_id, limit=5):
    """Get the most recently used profiles for a device."""
    device_profiles = mongo.db.device_profiles
    user_collection = mongo.db.users
    
    # Get the profiles associated with this device
    recent_profiles = device_profiles.find(
        {"device_id": device_id}
    )

    profiles = []
    for profile in recent_profiles:
        user = user_collection.find_one({"_id": ObjectId(profile["user_id"])})
        if user:
            user["_id"] = str(user["_id"])
            user.pop("password", None)  
            profiles.append(user)

    return profiles