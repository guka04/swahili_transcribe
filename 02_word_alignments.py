import os
from aeneas.executetask import ExecuteTask
from aeneas.task import Task

# Set up FFmpeg for Aeneas
os.environ["AENEAS_EXECUTE_TASK"] = "ffmpeg"

# Directories containing the audio files
data_dirs = [
    "/home/khaguh/swahili_bert_310/data/train",
    "/home/khaguh/swahili_bert_310/data/val",
    "/home/khaguh/swahili_bert_310/data/test"
]


def tokenize_text_file(text_path):
    """Tokenizes text file so that each word appears on a new line."""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    words = text.split()  # Tokenize by whitespace

    tokenized_text_path = text_path.replace(".txt", "_tokenized.txt")
    with open(tokenized_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))  # Save words line by line

    return tokenized_text_path


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
            else:
                print(f"Skipping {audio_path}: No corresponding text file found.")

print("All audio alignments completed successfully!")
