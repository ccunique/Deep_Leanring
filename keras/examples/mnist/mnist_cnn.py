#keras-gpu,2.0.5
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import  Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K

batch_size=128
num_class=10
epochs=12

#input image dimensions
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#X_train.shape=(60000,28,28),y_train_shape=60000,
img_rows,img_cols=28,28

#规范图片的表示格式使其与keras默认格式_IMAGE_DATA_FORMAT一致
if K.image_data_format() == 'channels_first':
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape=(1,img_rows,img_cols)
else: #'channels_last'
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)

#规范化
x_train = x_train/255
x_test =x_test/255
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

#one-hot encoding
y_train=keras.utils.to_categorical(y_train,num_class)
y_test=keras.utils.to_categorical(y_test,num_class)

#build cnn model
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test,verbose=0)
print("Test loss:",score[0])
print('Test accuracy',score[1])
