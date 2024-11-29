import parselmouth
from parselmouth.praat import call

AUDIO_DIR = 'audio_files'  # Update with your actual directory path
OUTPUT_CSV = 'voice_dataset.csv'

def extract_audio_features(file_path, gender):
    try:
        sound = parselmouth.Sound(file_path)
        
        # Set f0min and f0max based on gender
        if gender.lower() == "female":
            f0min = 100  # Minimum pitch frequency for females
            f0max = 300  # Maximum pitch frequency for females
        elif gender.lower() == "male":
            f0min = 75   # Minimum pitch frequency for males
            f0max = 200  # Maximum pitch frequency for males
        else:
            raise ValueError("Gender must be 'male' or 'female'.")

        # Fundamental frequency (Pitch) and standard deviation
        pitch = sound.to_pitch()
        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdevF0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        
        # Intensity and its standard deviation
        intensity = sound.to_intensity()
        meanIntensity = call(intensity, "Get mean", 0, 0)
        stdevIntensity = call(intensity, "Get standard deviation", 0, 0)
        
        # Harmonic Noise Ratio (HNR)
        harmonicity = sound.to_harmonicity()
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # # Jitter and RAP Jitter
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        # # Shimmer and APQ11 Shimmer
        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # # Phonation Time
        phonationTime = call(pointProcess, "Get total duration")  # Should only require the PointProcess
        
        # # Voice Turbulence Index (VTI)
        vti = sound.to_harmonicity()  # Use harmonicity for VTI
        meanVTI = call(vti, "Get mean", 0, 0)

        features = {
            "meanF0": meanF0,
            "stdevF0": stdevF0,
            "meanIntensity": meanIntensity,
            "stdevIntensity": stdevIntensity,
            "hnr": hnr,
            "localJitter": localJitter,
            "rapJitter": rapJitter,
            "localShimmer": localShimmer,
            "apq11Shimmer": apq11Shimmer,
            "phonationTime": phonationTime,
            "meanVTI": meanVTI
        }
        
        return features
    except Exception as e:
        raise Exception(f"Error extracting audio features: {e}")

