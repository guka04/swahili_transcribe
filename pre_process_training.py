import json
import os
import pandas as pd

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
            processed_data.append([text, start_time, end_time])
    
    return processed_data

def save_to_csv(json_folder, output_csv):
    all_data = []
    
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)
            all_data.extend(process_alignment_json(json_path))
    
    df = pd.DataFrame(all_data, columns=["Text", "Start_Time", "End_Time"])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved processed data to {output_csv}")

# Example usage
json_folder = "path/to/json/files"
output_csv = "processed_dataset.csv"
save_to_csv(json_folder, output_csv)
