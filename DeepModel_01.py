from tensorflow import keras
from keras import datasets, layers, models
from keras.layers import Conv2D, ReLU, Softmax, Flatten, BatchNormalization, MaxPooling2D, Dense, Dropout, Input
from keras.models import Sequential
import numpy as np

(x_train, y_train), (x_test,y_test) = datasets.mnist.load_data()

def createModel():
    model = Sequential()
    #model.add(Flatten(input_shape=(28,28,), name='Flat1'))
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(32,(3,3),activation = 'relu',name='Conv2'))
    model.add(MaxPooling2D((2,2),strides=2,name='MaxPool3'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation='relu',name='Conv5'))
    model.add(MaxPooling2D((2,2),strides=2,name='MaxPool6'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #model.add(Dense(1024,activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(10,activation='softmax'))  
    return model

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

y_test = keras.utils.to_categorical(y_test,10)
y_train = keras.utils.to_categorical(y_train,10)

print(x_train.shape)
print(y_train.shape)

batch_size = 128
epochs = 15

model = createModel()
model.build()
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train,y_train, batch_size=batch_size, epochs= epochs, validation_split=0.1)

score = model.evaluate(x_test,y_test,verbose=0)
print("Loss", score[0])
print("accuracy", score[1])



