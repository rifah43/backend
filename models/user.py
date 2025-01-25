from bson import ObjectId
from db import mongo
from marshmallow import Schema, fields, validate, ValidationError, validates
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

class UserSchema(Schema):
    id = fields.Str(dump_only=True)
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    password = fields.Str(required=True, load_only=True)
    age = fields.Int(required=True, validate=lambda n: 0 < n < 120)
    gender = fields.Str(required=True)
    height = fields.Float(required=True, validate=lambda n: 0 < n < 300)
    weight = fields.Float(required=True, validate=lambda n: 0 < n < 500)
    bmi = fields.Float(required=True)
    isActive = fields.Bool(default=False) 

    @validates('bmi')
    def validate_bmi(self, value):
        if not (15 <= value <= 40):
            raise ValidationError('BMI must be between 15 and 40')

class DeviceProfileSchema(Schema):
    device_id = fields.Str(required=True)
    user_id = fields.Str(required=True)
    isActive = fields.Bool(default=False)  

def add_user(user_data, device_id=None):
    schema = UserSchema()

    try:
        validated_data = schema.load(user_data)
    except ValidationError as err:
        return {"errors": err.messages}, 400

    try:
        user_collection = mongo.db.users
        device_profiles = mongo.db.device_profiles

        if user_collection.find_one({"email": validated_data["email"]}):
            return {"message": "User with this email already exists."}, 400

        validated_data["isActive"] = bool(device_id)
        validated_data["password"] = generate_password_hash(validated_data["password"])
        
        user_id = user_collection.insert_one(validated_data).inserted_id

        if device_id:
            # Update existing users and profiles
            device_profiles.update_many(
                {"device_id": device_id},
                {"$set": {"isActive": False}}
            )
            
            user_collection.update_many(
                {"_id": {"$ne": user_id}},
                {"$set": {"isActive": False}}
            )

            # Create new device profile
            device_profiles.insert_one({
                "device_id": device_id,
                "user_id": str(user_id),
                "isActive": True,
                "lastUsed": datetime.utcnow()
            })

        return {
            "message": "User created successfully",
            "user_id": str(user_id),
            "isActive": validated_data["isActive"]
        }, 201

    except Exception as e:
        print(f"Error in add_user: {str(e)}")
        return {
            "error": "Failed to create user",
            "details": str(e)
        }, 500

# Helper function to validate ObjectId
def is_valid_object_id(id_str):
    try:
        ObjectId(id_str)
        return True
    except:
        return False
    

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

def get_device_profiles(device_id):
    
    try:
        device_profiles = mongo.db.device_profiles
        user_collection = mongo.db.users
        
        # First, get all device profile IDs
        device_profile_cursor = device_profiles.find(
            {"device_id": device_id}
        ).sort([
            ("isActive", -1)  # Only sort by 'isActive' field
        ])
        
        # Convert cursor to list to avoid cursor timeout
        device_profile_list = list(device_profile_cursor)
        print(f"Found {len(device_profile_list)} device profiles for device {device_id}")  # Debug log
        
        profiles = []
        for profile in device_profile_list:
            try:
                # Validate user_id before querying
                if not profile.get("user_id"):
                    print(f"Warning: Device profile without user_id: {profile}")
                    continue
                    
                user = user_collection.find_one({"_id": ObjectId(profile["user_id"])})
                if user:
                    # Create a clean user profile object
                    user_profile = {
                        "id": str(user["_id"]),
                        "name": user.get("name", ""),
                        "email": user.get("email", ""),
                        "age": user.get("age"),
                        "gender": user.get("gender", ""),
                        "height": user.get("height"),
                        "weight": user.get("weight"),
                        "bmi": user.get("bmi"),
                        "isActive": profile.get("isActive", False)
                    }
                    profiles.append(user_profile)
                else:
                    print(f"Warning: User not found for ID: {profile['user_id']}")
            except Exception as e:
                print(f"Error processing user {profile.get('user_id')}: {str(e)}")
                continue
        
        # Return the profiles sorted only by 'isActive'
        return profiles
        
    except Exception as e:
        print(f"Error fetching device profiles: {str(e)}")
        raise ValueError(f"Failed to fetch device profiles: {str(e)}")
