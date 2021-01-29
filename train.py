import os

import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import adam

from Sound_processing import *

DIR = "Dataset"

temp = np.zeros((7,))
temp[0] = 1
y = []
size, x = Loading("Dataset/background_noise/")
for i in range(size):
	y.append(temp.copy())
size,xx = Loading("Dataset/down/")
x = np.concatenate((x,xx))
temp[0] = 0
temp[1] = 1
for i in range(size):
	y.append(temp.copy())
size,xx = Loading("Dataset/go/")
x = np.concatenate((x,xx))
temp[1] = 0
temp[2] = 1
for i in range(size):
	y.append(temp.copy())
size,xx = Loading("Dataset/left/")
x = np.concatenate((x,xx))
temp[2] = 0
temp[3] = 1
for i in range(size):
	y.append(temp.copy())
size,xx = Loading("Dataset/right/")
x = np.concatenate((x,xx))
temp[3] = 0
temp[4] = 1
for i in range(size):
	y.append(temp.copy())
size,xx = Loading("Dataset/stop/")
x = np.concatenate((x,xx))
temp[4] = 0
temp[5] = 1
for i in range(size):
	y.append(temp.copy())
size,xx = Loading("Dataset/up/")
x = np.concatenate((x,xx))
temp[5] = 0
temp[6] = 1
for i in range(size):
	y.append(temp.copy())


print(x.shape)
x = np.resize(x,(len(y),177,98))
y = np.asarray(y)


shape = (177,98,)
model = Sequential()
model.add(CuDNNLSTM(128,input_shape = shape,return_sequences = True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer = adam(0.0002),loss = 'mse',metrics = ['accuracy'])

print(y[0])

model.load_weights("Model.h5")

while (True):
	model.fit(x,y,epochs = 50,verbose = 2)
	model.save_weights("Model.h5")

