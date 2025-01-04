import os
import parselmouth
from parselmouth.praat import call
import numpy as np
import tempfile

def extract_audio_features(file_storage, gender):
    """
    Extract acoustic features from uploaded audio file
    
    Args:
        file_storage: FileStorage object from Flask
        gender: 'male' or 'female'
    """
    try:
        # Save the uploaded file to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'temp.wav')
        
        file_storage.save(temp_path)
        
        # Set pitch range based on gender
        f0min = 75 if gender.lower() == 'male' else 100
        f0max = 200 if gender.lower() == 'male' else 300
        
        # Create Praat sound object from saved file
        sound = parselmouth.Sound(temp_path)
        
        # Extract pitch features
        pitch = sound.to_pitch()
        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdevF0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        
        # Extract intensity features
        intensity = sound.to_intensity()
        meanInten = call(intensity, "Get mean", 0, 0)
        stdevInten = call(intensity, "Get standard deviation", 0, 0)
        
        # Extract harmonicity
        harmonicity = sound.to_harmonicity()
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # Extract jitter and shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        
        localJitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        localShimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Clean up temporary file
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        features = {
            'meanF0': float(meanF0),
            'stdevF0': float(stdevF0),
            'meanInten': float(meanInten),
            'stdevInten': float(stdevInten),
            'hnr': float(hnr),
            'localJitter': float(localJitter),
            'rapJitter': float(rapJitter),
            'localShimmer': float(localShimmer),
            'apq11Shimmer': float(apq11Shimmer)
        }
        
        # Validate extracted features
        if any(np.isnan(value) for value in features.values()):
            raise ValueError("Invalid audio features detected (NaN values)")
            
        return features
        
    except Exception as e:
        raise Exception(f"Error extracting audio features: {str(e)}")
    finally:
        # Ensure temporary files are cleaned up
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)