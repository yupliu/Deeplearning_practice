import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
import tensorflow as tf
from tensorflow import keras
from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

mist = tf.keras.datasets.fashion_mnist
(train_data, train_label), (test_data, test_label) = mist.load_data()

plt.figure()
plt.imshow(train_data[0,])
plt.colorbar()
plt.grid(False)
plt.show()

train_data = train_data / 255.0
test_data = test_data / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i,], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_label[i]])
plt.show()

# Equivalent
# model = tf.keras.Sequential(name='image_process')
# model.add(layers.Flatten(input_shape=(28, 28), name='layer1'))
# model.add(layers.Dense(128, activation='relu', name='layer2'))
# model.add(layers.Dense(10, name='layer3'))


# Equivalent
# model = tf.keras.Sequential([
#    tf.keras.layers.Flatten(input_shape=(28, 28)),
#    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.Dense(10)
# ]
# )


def seq_model(in_shape):
    x_input = Input(in_shape)
    x = Flatten()(x_input)
    x = Dense(128, activation='relu')(x)
    x = Dense(10)(x)
    model_1 = Model(inputs=x_input, outputs=x)
    return model_1


model = seq_model((28, 28))
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_data, train_label, epochs=10)

test_loss, test_acc = model.evaluate(test_data, test_label, verbose=2)
print('accuracy=', test_acc)

#prediction = model.predict(test_data)
#generate 10 dimension value need to call softmax to convert to probability

predict_model = tf.keras.Sequential([model, layers.Softmax()])
prediction = predict_model.predict(test_data)
predict_class = np.argmax(pred, axis=1)
predict_class = predict_class.tolist()
predict_class

