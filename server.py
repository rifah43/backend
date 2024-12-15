from flask import Flask
from routes.featureExtractionRealtime import extract_features_blueprint
from routes.userRoutes import user_blueprint
from routes.trendRoutes import trends
from dotenv import load_dotenv
import os
from db import init_app, mongo  

load_dotenv()

app = Flask(__name__)

app.config["MONGO_URI"] = f"{os.getenv('FLASK_MONGO')}/AcoustiCare"
app.config['SECRET_KEY'] = f"{os.getenv('SECRET_KEY')}"

init_app(app)

if mongo.db is None:
    print("MongoDB not initialized")
else:
    print("MongoDB initialized successfully")

app.register_blueprint(extract_features_blueprint)
app.register_blueprint(user_blueprint)  
app.register_blueprint(trends)  

if __name__ == '__main__':
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
