import librosa
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from sklearn.preprocessing import LabelBinarizer  # For one-hot encoding


def load_data(csv_file, audio_dir, global_max_len=None): # Add global_max_len
    df = pd.read_csv(csv_file)
    X = []
    y = []
    max_len = 0
    audio_cache = {}  # Store loaded audio files

    # Calculate global max_len for the first directory
    max_len = 0  # Initialize max_len
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['audio_filename'])
        start_time = row['Start_Time']
        end_time = row['End_Time']
        audio_signal, sr = librosa.load(audio_path, sr=None)
        audio_segment = audio_signal[int(start_time * sr):int(end_time * sr)]
        mel_spectrogram = get_mel_spectrogram(audio_segment, sr)
        max_len = max(max_len, mel_spectrogram.shape[1])  # Update max_len

       
    # Now process the data and pad using global max_len

       
    # Now process the data and pad using global max_len
    for index, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['audio_filename'])
        
        # Load the audio file and add it to the cache
        if audio_path not in audio_cache:
            audio_signal, sr = librosa.load(audio_path, sr=None)
            audio_cache[audio_path] = (audio_signal, sr)
        else:
            audio_signal, sr = audio_cache[audio_path]

        audio_signal, sr = audio_cache[audio_path]  # Retrieve from cache
        start_time = row['Start_Time']
        end_time = row['End_Time']
        audio_segment = audio_signal[int(start_time * sr):int(end_time * sr)]
        mel_spectrogram = get_mel_spectrogram(audio_segment, sr)

        # Pad the spectrogram using global max_len
        pad_width = max_len - mel_spectrogram.shape[1]
        if pad_width > 0:
            padded_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_spectrogram = mel_spectrogram

        X.append(padded_spectrogram)
        y.append(row['Text'])

        return np.array(X), np.array(y), max_len  # Add max_len to the return statement

def get_mel_spectrogram(audio_segment, sr):
       mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_fft=512)  # Reduced n_fft
       mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
       return mel_spectrogram_db

# Directories containing the audio/json/csv files
data_dirs = [
    "/home/khaguh/swahili_bert_310/data/train"
    # "/home/khaguh/swahili_bert_310/data/val",
    # "/home/khaguh/swahili_bert_310/data/test"
]

# Load and combine data from all directories
X_all = []
y_all = []
global_max_len = 0 # Initialize

for data_dir in data_dirs:
    dir_name = os.path.basename(data_dir)
    output_csv = os.path.join(data_dir, dir_name + "_processed.csv")
    X, y, max_len = load_data(output_csv, data_dir) # get max_len
    X_all.extend(X)
    y_all.extend(y)
    global_max_len = max(global_max_len, max_len) # update global_max_len
    

# Re-pad and reload all data using global_max_len
X_all = []
y_all = []
for data_dir in data_dirs:
    dir_name = os.path.basename(data_dir)
    output_csv = os.path.join(data_dir, dir_name + "_processed.csv")
    X, y, _ = load_data(output_csv, data_dir, global_max_len) # provide global_max_len
    X_all.extend(X)
    y_all.extend(y)

X_all = np.array(X_all)
y_all = np.array(y_all)


# Split data
# Adjust test_size to ensure non-empty training set
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Reshape data for LSTM
# Assuming your Mel Spectrograms have shape (num_frames, num_features)
# Reshape to (num_samples, num_timesteps, num_features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) 
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# One-hot encode targets
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
vocab_size = len(encoder.classes_)  # Get the vocabulary size
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)

# One-hot encode targets
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
vocab_size = len(encoder.classes_)  # Get the vocabulary size
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)

# Model building
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))  
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))  

# ... (model compiling, training, and evaluation)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# predictions = model.predict(X_test)  
# Process predictions to get transcribed text (e.g., using a character-to-text mapping)
predictions = model.predict(X_test)

transcribed_texts = []
for prediction in predictions:
    predicted_indices = np.argmax(prediction, axis=1)  # Get most likely indices
    predicted_chars = encoder.classes_[predicted_indices]  # Map indices to characters/words
    transcribed_text = ''.join(predicted_chars)  # Join characters/words
    transcribed_texts.append(transcribed_text)

# Print the transcribed texts
for text in transcribed_texts:
    print(text)  
