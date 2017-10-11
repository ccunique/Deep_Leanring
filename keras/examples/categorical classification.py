# keras-gpu,2.0.5
from keras.models import Sequential
from keras.layers import Dense,Activation
import keras

model=Sequential()
model.add(Dense(units=64,activation='relu',input_dim=100))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

import numpy as np
data=np.random.random((1000,100))
labels=np.random.randint(10,size=(1000,1))

from sklearn import preprocessing
min_max_scaler=preprocessing.MinMaxScaler()
data_scaler=min_max_scaler.fit_transform(data)

#Convert labels to categorical one-hot encoding
one_hot_labels=keras.utils.to_categorical(labels,num_classes=10)

model.fit(data_scaler,one_hot_labels,epochs=10,batch_size=128)
