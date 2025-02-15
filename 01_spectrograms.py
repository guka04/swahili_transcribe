import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # Load audio
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S_dB, sr

# Example usage
tweet_210_wav = "/home/khaguh/swahili_bert/data/tweet_210/tweet_210.wav"
spectrogram, sr = audio_to_spectrogram(tweet_210_wav)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.show()
