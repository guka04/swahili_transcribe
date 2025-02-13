from transformers import AutoTokenizer
import pandas as pd


tweet_210_txt = "/home/khaguh/swahili_bert/data/tweet_210/tweet_210.txt"
# Load pre-trained BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# Define Swahili-specific vocabulary
swahili_vocab = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "r", "s", "t", "u", "v", "w", "y", "z", " ", "'"
]

# Create vocab_dict (no need to re-train tokenizer)
# vocab_dict = {char: idx for idx, char in enumerate(swahili_vocab)}
# df = pd.read_csv(tweet_210_txt,sep='\t')

# Update processor for Swahili vocabulary
# tokenizer.add_tokens(swahili_vocab)

with open(tweet_210_txt, "r") as file:
            sentence = file.read().strip()
print(sentence)           

tokens = tokenizer.tokenize(sentence)
print(tokens)