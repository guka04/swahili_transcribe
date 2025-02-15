import os
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import pandas as pd
import re

# Set up FFmpeg for Aeneas
os.environ["AENEAS_EXECUTE_TASK"] = "ffmpeg"

# Directories containing the audio files
data_dirs = [
    "/home/khaguh/swahili_bert_310/data/train"
    # "/home/khaguh/swahili_bert_310/data/val",
    # "/home/khaguh/swahili_bert_310/data/test"
]



def tokenize_text_file(text_path):
    """Tokenizes text file so that each word appears on a new line."""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

        # Clean up the text by removing unwanted tokens
        text = text.replace("<s>", "").replace("</s>", "").replace("< s>", "").replace("< /s>", "").strip()
        text = re.sub(r'\(tweet_\d+\)', '', text)  # Remove (tweet_XXXX) pattern
        text = " ".join(text.split())  # Remove any extra spaces

    print(text)
    words = text.split()  # Tokenize by whitespace

    tokenized_text_path = text_path.replace(".txt", "_tokenized.txt")
    with open(tokenized_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))  # Save words line by line

    return tokenized_text_path

def process_alignment_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fragments = data.get("fragments", [])
    processed_data = []
    
    for fragment in fragments:
        if "lines" in fragment:
            text = " ".join(fragment["lines"])
            start_time = float(fragment["begin"])
            end_time = float(fragment["end"])

            # Skip fragments where start and end times are the same or too close
            if start_time == end_time or abs(start_time - end_time) < 0.01:
                continue

            # Manually fix segmentation errors
            text = " ".join([fix_fragment_tokens(word) for word in text.split()])
            processed_data.append([text, start_time, end_time])
    
    return processed_data

def fix_fragment_tokens(fragment):
    # Manually fix common segmentation errors
    if fragment in ["kal", "a-azar"]:
        return "kala-azar"
    return fragment

def update_language_in_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        alignment = json.load(f)

    # Update the language tag to Swahili
    alignment['language'] = 'sw'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(alignment, f, ensure_ascii=False, indent=4)

def align_audio(audio_path, text_path, json_path):
    """Aligns audio with tokenized text using Aeneas and saves alignment as JSON."""
    tokenized_text_path = tokenize_text_file(text_path)  # Ensure word-level alignment

    config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"

    task = Task(config_string=config_string)
    task.audio_file_path_absolute = audio_path
    task.text_file_path_absolute = tokenized_text_path  # Use tokenized version
    task.sync_map_file_path_absolute = json_path

    print(f"Aligning: {audio_path} -> {json_path}")

    ExecuteTask(task).execute()
    task.output_sync_map_file()


def plot_spectrogram_with_alignment(audio_path, json_path):
    """Plots the spectrogram and overlays the word alignment timestamps."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Load alignment JSON
    with open(json_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)

    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram with Word Alignment ({os.path.basename(audio_path)})")

    # Overlay word timestamps
    for fragment in alignment["fragments"]:
        start = float(fragment["begin"])
        word = fragment["lines"][0]  # Extract the word from "lines" array
        plt.axvline(x=start, color='r', linestyle='--', alpha=0.7)
        plt.text(start, S.shape[0] - 10, word, color="white", fontsize=10, rotation=45)

    plt.savefig(f"{json_path.replace('.json', '.png')}", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

# def process_alignment_json(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     fragments = data.get("fragments", [])
#     processed_data = []
    
#     for fragment in fragments:
#         if "lines" in fragment:
#             text = " ".join(fragment["lines"])
#             start_time = float(fragment["begin"])
#             end_time = float(fragment["end"])
#             processed_data.append([text, start_time, end_time])
    
#     return processed_data

def save_to_csv(json_folder, output_csv):
    all_data = []
    
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)
            audio_filename = "trimmed_"+ filename.replace(".json", ".wav")  # Assuming audio is .wav 
            processed_data = process_alignment_json(json_path)
            
            # Add audio filename to each segment
            for segment in processed_data:
                segment.insert(0, audio_filename)  # Insert at the beginning
                
            all_data.extend(processed_data)
    
    df = pd.DataFrame(all_data, columns=["audio_filename", "Text", "Start_Time", "End_Time"])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved processed data to {output_csv}")


# Process each directory
for data_dir in data_dirs:
    for filename in os.listdir(data_dir):
        if filename.startswith("trimmed_") and filename.endswith(".wav"):
            base_name = filename.replace("trimmed_", "").replace(".wav", "")

            audio_path = os.path.join(data_dir, filename)
            text_path = os.path.join(data_dir, base_name + ".txt")
            json_path = os.path.join(data_dir, base_name + ".json")

            # Check if corresponding text file exists before alignment
            if os.path.exists(text_path):
                align_audio(audio_path, text_path, json_path)
                update_language_in_json(json_path)
                plot_spectrogram_with_alignment(audio_path, json_path)  # 🎶📊 Plot visualization
            else:
                print(f"Skipping {audio_path}: No corresponding text file found.")

print("All audio alignments completed successfully!")

for data_dir in data_dirs:
    dir_name = os.path.basename(data_dir)
    output_csv = os.path.join(data_dir, dir_name + "_processed.csv")
    save_to_csv(data_dir, output_csv)
