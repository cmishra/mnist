from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.regularizers import l2
from source import DataNormalizer, plot_performance
from matplotlib import pyplot as plt


# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(np.unique(y_test))
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
normalizer = DataNormalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)

init_scheme = 'he_normal'
num_runs = 100
nb_epoch = 5
l = 9.5 * 10E-5

w_regularizer = l2(l)
model = Sequential()
model.add(Flatten(input_shape=((28,28))))
model.add(Dense(50, input_dim=28*28, activation='relu', init=init_scheme,
                W_regularizer=w_regularizer))
model.add(Dense(50, activation='relu', init=init_scheme,
                W_regularizer=w_regularizer))
model.add(Dense(10, activation='softmax', init=init_scheme,
                W_regularizer=w_regularizer))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
hist = model.fit(x_train, y_train, batch_size=256, nb_epoch=nb_epoch,
                validation_split=0.2,
                 show_accuracy=True)

w_regularizer = l2(2878.33400367)
model = Sequential()
model.add(Flatten(input_shape=((28,28))))
model.add(Dense(50, input_dim=28*28, activation='relu', init=init_scheme,
                W_regularizer=w_regularizer))
model.add(Dense(50, activation='relu', init=init_scheme,
                W_regularizer=w_regularizer))
model.add(Dense(10, activation='softmax', init=init_scheme,
                W_regularizer=w_regularizer))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
hist2 = model.fit(x_train, y_train, batch_size=256, nb_epoch=nb_epoch,
                validation_split=0.2,
                 show_accuracy=True)
plt.subplot(211)
plot_performance(hist, hist2, 'acc')
plt.subplot(212)
plot_performance(hist, hist2, 'acc')
plt.show()