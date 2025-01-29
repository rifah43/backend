import os
from pydub import AudioSegment
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class AudioAugmentation:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.supported_formats = ('.wav', '.mp3', '.m4a', '.opus')

    def augment_audio(self, audio, filename):
        augmented_versions = []

        # Apply various augmentations
        augmented_versions.append((audio + 6, f"{filename}_increased_volume"))
        augmented_versions.append((audio - 6, f"{filename}_decreased_volume"))
        augmented_versions.append((audio.speedup(playback_speed=1.5), f"{filename}_speedup"))
        augmented_versions.append((audio.speedup(playback_speed=0.7), f"{filename}_slowdown"))

        return augmented_versions

    def process_audio_files(self):
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(self.supported_formats):
                file_path = os.path.join(self.input_dir, filename)
                output_file_prefix = os.path.splitext(filename)[0]

                try:
                    audio = AudioSegment.from_file(file_path)
                    augmented_versions = self.augment_audio(audio, output_file_prefix)

                    # Save augmented versions
                    for augmented_audio, augmented_name in augmented_versions:
                        output_file = os.path.join(self.output_dir, f"{augmented_name}.wav")
                        augmented_audio.export(output_file, format="wav")
                        print(f"Saved: {output_file}")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    AUDIO_DIR = Config.DATASET_DIR
    OUTPUT_DIR = os.path.join(AUDIO_DIR, "augmented")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Augmenting audio files from {AUDIO_DIR} and saving to {OUTPUT_DIR}")

    augmenter = AudioAugmentation(input_dir=AUDIO_DIR, output_dir=OUTPUT_DIR)
    augmenter.process_audio_files()

    print("Audio augmentation complete.")
