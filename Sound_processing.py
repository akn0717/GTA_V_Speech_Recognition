#this code belongs to dawidkopczyk from his github speech recognition
#repo link: https://github.com/dawidkopczyk/speech_recognition


from scipy.io import wavfile
from scipy.signal import stft
import numpy as np
import os

def read_wav_file(x):
    # Read wavfile using scipy wavfile.read
    _, wav = wavfile.read(x) 
    # Normalize
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            
    return wav

def read_wave(wav):
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

def process_wav(x, threshold_freq=5500, eps=1e-10):
    # Read wav file to array
    wav = read_wave(x) #read_wav_file(x)
    # Sample rate
    L = 16000
    # If longer then randomly truncate
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]  
    # If shorter then randomly add silence
    elif len(wav) < L:
        rem_len = L - len(wav)
        silence_part = np.random.randint(-100,100,16000).astype(np.float32) / np.iinfo(np.int16).max
        j = np.random.randint(0, rem_len)
        silence_part_left  = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    # Create spectrogram using discrete FFT (change basis to frequencies)
    freqs, times, spec = stft(wav, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
    # Cut high frequencies
    if threshold_freq is not None:
        spec = spec[freqs <= threshold_freq,:]
        freqs = freqs[freqs <= threshold_freq]
    # Log spectrogram
    amp = np.log(np.abs(spec)+eps)

    return np.expand_dims(amp, axis=2) 

def process_wav_file(x, threshold_freq=5500, eps=1e-10):
    # Read wav file to array
    wav = read_wav_file(x)
    # Sample rate
    L = 16000
    # If longer then randomly truncate
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]  
    # If shorter then randomly add silence
    elif len(wav) < L:
        rem_len = L - len(wav)
        silence_part = np.random.randint(-100,100,16000).astype(np.float32) / np.iinfo(np.int16).max
        j = np.random.randint(0, rem_len)
        silence_part_left  = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    # Create spectrogram using discrete FFT (change basis to frequencies)
    freqs, times, spec = stft(wav, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
    # Cut high frequencies
    if threshold_freq is not None:
        spec = spec[freqs <= threshold_freq,:]
        freqs = freqs[freqs <= threshold_freq]
    # Log spectrogram
    amp = np.log(np.abs(spec)+eps)

    return np.expand_dims(amp, axis=2) 


def Loading(path):
    data_dir = path
    x_train = []
    img_file = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    for i in img_file:
        x_train.append(process_wav_file(data_dir + i))
    return len(x_train),np.asarray(x_train)