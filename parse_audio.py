import tensorflow as tf;
import tensorflow_io as tfio
import os
from matplotlib import pyplot as plt

#print(tf.reduce_sum(tf.random.normal([1000, 1000])))
tweet_210_wav = "/home/khaguh/swahili_bert/data/tweet_210/tweet_210.wav"
tweet_210_txt = "/home/khaguh/swahili_bert/data/tweet_210/tweet_210.txt"

# Ensuring directories are accessibleclear

assert os.path.exists(tweet_210_wav), "Wav file not found."
assert os.path.exists(tweet_210_txt), "Transcript file not found."

#Goes from 44100Hz to 16000hz
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


tweet_210_wave = load_wav_16k_mono(tweet_210_wav)

plt.plot(tweet_210_wave)
#plt.plot(tweet_210_wave)
plt.show()
# print(len(tweet_210_wave))
# end=len(tweet_210_wave)-70000
# start=70000
# tweet_210_wave_2=tweet_210_wave[start,end]
# plt.plot(tweet_210_wave_2)
# # #plt.plot(tweet_210_wave)
# plt.show()

# def preprocess(file_path, label): 
#     wav = load_wav_16k_mono(file_path)
#     length= 48000 #len(wav)
#     wav = wav[:length]
#     zero_padding = tf.zeros([length] - tf.shape(wav), dtype=tf.float32)
#     wav = tf.concat([zero_padding, wav],0)
#     spectrogram = tf.signal.stft(wav, frame_length=16, frame_step=8)
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.expand_dims(spectrogram, axis=2)
#     return spectrogram, label

def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


spectrogram, label = preprocess(tweet_210_wav, 1)
print(spectrogram)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()

