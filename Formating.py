import os

from scipy.io import wavfile
from scipy.signal import stft

_, wav = wavfile.read("output.wav") 

print(wav)