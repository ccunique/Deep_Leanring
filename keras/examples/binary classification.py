# keras-gpu,2.0.5
from keras.models import Sequential
from keras.layers import Dense,Activation

model=Sequential()

model.add(Dense(32,activation='relu',input_dim=100)) #output is 32dim
model.add(Dense(1,activation='sigmoid'))    #output is 1dim,binary classification

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

import numpy as np
data=np.random.random((1000,100))
labels=np.random.randint(2,size=(1000,1))

model.fit(data,labels,epochs=10,batch_size=32)
