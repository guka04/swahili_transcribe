import os
tweet_210_wav = "/home/khaguh/swahili_bert_310/data/tweet_210/fixed_tweet_210.wav"
tweet_210_txt = "/home/khaguh/swahili_bert_310/data/tweet_210/tweet_210.txt"
tweet_210_json = "/home/khaguh/swahili_bert_310/data/tweet_210/alignment.json"


if not os.path.exists(tweet_210_wav):
    print(f"Error: {tweet_210_wav} does not exist!")
else:
    print(f"Success: {tweet_210_wav} found!")


with open(json_path, 'r', encoding='utf-8') as f:
    alignment = json.load(f)

# Update language tag if necessary
alignment['language'] = 'sw'

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(alignment, f, ensure_ascii=False, indent=4)
