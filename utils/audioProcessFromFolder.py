import os
import re
import pandas as pd
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
from config import Config

class VoiceFeatureExtractor:
   def __init__(self):
       """Initialize the feature extractor"""
       self.features = {}
       
   def parse_filename(self, filename):
       """
       Parse participant info from filename 
       Format: age_height_weight_gender_diagnosis_name (optional_number)
       Example: 25_68_69_male_Nondiabetic_mahidi (73)
       """
       try:
           pattern = r"(\d+)_(\d+)_(\d+)_(male|female)_(T2DM|Nondiabetic)"
           match = re.match(pattern, filename)
           
           if not match:
               raise ValueError(f"Invalid filename format: {filename}")
               
           age = int(match.group(1))
           height_inches = int(match.group(2))
           weight_kg = int(match.group(3)) 
           gender = match.group(4).lower()
           diagnosis = match.group(5)
           
           # Calculate BMI: weight(kg) / height(m)²
           height_m = height_inches * 0.0254
           bmi = round(weight_kg / (height_m * height_m), 2)
           
           return {
               "Age": age, 
               "BMI": bmi,
               "Gender": gender,
               "Diagnosis": diagnosis
           }
           
       except Exception as e:
           raise Exception(f"Error parsing filename {filename}: {str(e)}")

   def extract_features(self, file_path, gender):
       """Extract acoustic features from voice recording"""
       try:
           # Convert audio to wav if needed
           if not file_path.endswith('.wav'):
               audio = AudioSegment.from_file(file_path)
               file_path = self._convert_to_wav(audio)

           sound = parselmouth.Sound(file_path)
           
           # Set pitch range based on gender
           f0min = 100 if gender == "female" else 75
           f0max = 300 if gender == "female" else 200
           
           # Extract pitch features
           pitch = sound.to_pitch()
           self.features['meanF0'] = call(pitch, "Get mean", 0, 0, "Hertz") 
           self.features['stdevF0'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
           
           # Extract intensity features  
           intensity = sound.to_intensity()
           self.features['meanInten'] = call(intensity, "Get mean", 0, 0)
           self.features['stdevInten'] = call(intensity, "Get standard deviation", 0, 0)
           
           # Extract harmonicity (HNR)
           harmonicity = sound.to_harmonicity()
           self.features['HNR'] = call(harmonicity, "Get mean", 0, 0)
           
           # Extract jitter and shimmer variations
           point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
           
           # Jitter measures
           self.features['localJitter'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
           self.features['rapJitter'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
           self.features['ppq5Jitter'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
           
           # Shimmer measures
           self.features['localShimmer'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
           self.features['localDbShimmer'] = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
           self.features['apq3Shimmer'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
           self.features['apq5Shimmer'] = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
           self.features['apq11Shimmer'] = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
           
           # Clean up temporary WAV file if created
           if not file_path.endswith('.wav'):
               os.remove(file_path)
               
           return self.features

       except Exception as e:
           raise Exception(f"Error extracting features: {str(e)}")
           
   def _convert_to_wav(self, audio):
       """Convert audio to WAV format"""
       temp_path = "temp.wav"
       audio.export(temp_path, format="wav")
       return temp_path

def process_audio_folder(folder_path, output_csv):
   """Process all audio files in a folder"""
   extractor = VoiceFeatureExtractor()
   all_features = []
   participant_id = 1

   for filename in os.listdir(folder_path):
       if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.opus')):
           file_path = os.path.join(folder_path, filename)
           
           try:
               # Parse participant info from filename
               participant_info = extractor.parse_filename(filename)
               
               # Extract voice features
               voice_features = extractor.extract_features(file_path, participant_info['Gender'])
               
               # Combine all features
               features = {
                   'ParticipantID': participant_id,
                   'Age': participant_info['Age'],
                   'BMI': participant_info['BMI'], 
                   'Gender': participant_info['Gender'],
                   'Diagnosis': participant_info['Diagnosis']
               }
               features.update(voice_features)
               
               all_features.append(features)
               participant_id += 1
               print(f"Processed {filename}")
               
           except Exception as e:
               print(f"Error processing {filename}: {str(e)}")
               continue

   # Create DataFrame with specific column order
   columns = ['ParticipantID', 'Age', 'BMI', 'Gender', 'Diagnosis', 
              'meanF0', 'stdevF0', 'meanInten', 'stdevInten', 'HNR',
              'localShimmer', 'localDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer',
              'localJitter', 'rapJitter', 'ppq5Jitter']
              
   df = pd.DataFrame(all_features)
   df = df[columns]  # Reorder columns
   df.to_csv(output_csv, index=False)
   print(f"Features saved to {output_csv}")

if __name__ == "__main__":
   AUDIO_DIR = Config.DATASET_DIR
   OUTPUT_CSV = os.path.join(AUDIO_DIR, "dataset.csv")
   print(f"Processing files in folder: {AUDIO_DIR}")
   process_audio_folder(AUDIO_DIR, OUTPUT_CSV)
   print(f"Feature extraction complete. Results saved to: {OUTPUT_CSV}")