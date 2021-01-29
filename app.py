import pyaudio
import wave
import numpy as np

from directkeys import PressKey, W, S, A, D, ReleaseKey, NP_6, NP_9
from Sound_processing import *
from collections import deque


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import adam

CHUNK = 1024*20
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
delay = 5
RECORD_SECONDS = 0.2
WAVE_OUTPUT_FILENAME = "output.wav"

work = []

drive = False

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

frames = deque(maxlen = int(RATE / CHUNK * RECORD_SECONDS))


shape = (177,98,)
model = Sequential()
model.add(CuDNNLSTM(128,input_shape = shape,return_sequences = True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(7,activation='softmax'))
model.compile(optimizer = adam(0.00002,decay = 1e-5),loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.load_weights("Model.h5")

def get_max(x):
	m = 0
	for i in range(len(x[0])):
		if (x[0,m]<x[0,i]):
			m = i
	return m

def category(x):
	if (x==0):
		return "background_noise"
	elif (x==1):
		return "Down"
	elif (x==2):
		return "Go"
	elif (x==3):
		return "Left"
	elif (x==4):
		return "Right"
	elif (x==5):
		return "Stop"
	else:
		return "Up"

def simulate(x):
	if (x==0):
		return
	if (x==2):
		PressKey(W)
		return
	elif (x==5):
		ReleaseKey(W)
		return
	if (x==1):
		PressKey(NP_6)
	elif (x==2):
		PressKey(W)
	elif (x==3):
		PressKey(A)
	elif (x==4):
		PressKey(D)
	elif (x==5):
		ReleaseKey(W)
	elif (x==6):
		PressKey(NP_9)
	work.append([x,delay])

def stop(x):
	if (x==1):
		ReleaseKey(NP_6)
	elif (x==3):
		ReleaseKey(A)
	elif (x==4):
		ReleaseKey(D)
	elif (x==5):
		ReleaseKey(W)
	elif (x==6):
		ReleaseKey(NP_9)

print("* recording")


while (True):
    data = np.fromstring(stream.read(CHUNK),np.int16)
    x_test = np.resize(process_wav(data),(1,177,98))
    x = model.predict(x_test)
    answer = get_max(x)
    simulate(answer)
    for i in range(len(work)):
    	work[i][1]-=1
    	if (work[i][1]<=0):
    		stop(work[i][0])
    		work.pop(i)
    		break
    print(category(answer))

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()