from flask import Blueprint, request, jsonify
import os
import pandas as pd
from utils.audioProcessing import extract_audio_features
from config import TEMP_DIR
import wave

def is_wav_file(file_path):
    try:
        with wave.open(file_path, 'rb') as f:
            return True
    except wave.Error as e:
        print(f"File is not a valid WAV file: {e}")
        return False

extract_features_blueprint = Blueprint('extract_features', __name__)

@extract_features_blueprint.route('/extract-features', methods=['POST'])
def extract_features():

    # Check if the audio file is in the request
    if 'audioFile' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audioFile']
    
    file_path = os.path.join(TEMP_DIR, file.filename)

    try:
        file.save(file_path)
    
    # Validate WAV file
        if not is_wav_file(file_path):
            return jsonify({"error": "Uploaded file is not a valid WAV file."}), 410
        features = extract_audio_features(file_path, "female")
        return jsonify(features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return jsonify({"error": str(e)}), 500

