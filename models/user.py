from db import mongo
from marshmallow import Schema, fields, validate, ValidationError


class UserSchema(Schema):
    name = fields.Str(required=True)
    age = fields.Int(required=True, validate=validate.Range(min=0))
    gender = fields.Str(required=True, validate=validate.OneOf(['Male','Female','Other']))
    email = fields.Email(required=True)
    height = fields.Float(required=True)
    weight = fields.Float(required=True)
    bmi = fields.Float(required=True)
    isActive = fields.Boolean( missing=False)

def add_user(user_data):
    schema = UserSchema()

    try:
        validated_data = schema.load(user_data)
    except ValidationError as err:
        return {"errors": err.messages}, 400

    try:
        user_collection = mongo.db.get_collection('users')
    except Exception as e:
        raise ValueError(f"Error accessing the collection: {e}")

    # Check for duplicate name and email
    existing_user = user_collection.find_one({
        "name": validated_data["name"],
        "email": validated_data["email"]
    })
    
    if existing_user:
        return {"message": "User with this name and email pair already exists."}, 400

    user_id = user_collection.insert_one(validated_data).inserted_id
    return {"message": "User created", "user_id": str(user_id)}, 201
