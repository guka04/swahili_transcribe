from aeneas.executetask import ExecuteTask
from aeneas.task import Task

# Define the configuration string
config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"

# Create a Task object with the config string
task = Task(config_string=config_string)

# Set absolute file paths
task.audio_file_path_absolute = "/home/khaguh/swahili_bert_310/data/tweet_210/fixed_tweet_210.wav"
task.text_file_path_absolute = "/home/khaguh/swahili_bert_310/data/tweet_210/tweet_210.txt"
task.sync_map_file_path_absolute = "/home/khaguh/swahili_bert_310/data/tweet_210/alignment.json"

# Print paths to verify
print("Audio Path:", task.audio_file_path_absolute)
print("Text Path:", task.text_file_path_absolute)
print("Sync Map Path:", task.sync_map_file_path_absolute)

# Execute task
ExecuteTask(task).execute()

# Save the sync map
task.output_sync_map_file()
