import librosa
import soundfile as sf
import os

# Directories containing the audio files
data_dirs = [
    "/home/khaguh/swahili_bert_310/data/train",
    "/home/khaguh/swahili_bert_310/data/val",
    "/home/khaguh/swahili_bert_310/data/test"
]

def trim_audio(audio_path, output_path):
    """Load audio, trim silence, and save the output."""
    y, sr = librosa.load(audio_path, sr=None)  # Load audio file
    y_trimmed, _ = librosa.effects.trim(y)  # Trim silence
    sf.write(output_path, y_trimmed, sr)  # Save trimmed audio

def process_directory(directory):
    """Loop through all WAV files in a directory and trim them."""
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):  # Process only .wav files
            audio_path = os.path.join(directory, filename)
            trimmed_audio_path = os.path.join(directory, "trimmed_" + filename)
            
            print(f"Trimming: {audio_path} -> {trimmed_audio_path}")
            trim_audio(audio_path, trimmed_audio_path)

# Process all directories
for data_dir in data_dirs:
    process_directory(data_dir)

print("All audio files processed successfully!")
