import tensorflow as tf;
import os
from matplotlib import pyplot as plt

#print(tf.reduce_sum(tf.random.normal([1000, 1000])))
tweet_210_wav = "/home/khaguh/swahili_bert/data/tweet_210/tweet_210.wav"
tweet_210_txt = "/home/khaguh/swahili_bert/data/tweet_210/tweet_210.txt"

# Ensuring directories are accessible
assert os.path.exists(tweet_210_wav), "Wav file not found."
assert os.path.exists(tweet_210_txt), "Transcript file not found."

