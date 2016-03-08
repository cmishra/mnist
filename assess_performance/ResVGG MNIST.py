from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from source import DataNormalizer, get_activations
from residual_blocks import building_residual_block

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(np.unique(y_test))
y_train_1dim = y_train
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
normalizer = DataNormalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[1]))
print(x_train.shape)

model = Sequential()
model.add(building_residual_block('res1', (1, 28, 28), 32, (3, 3), subsample=(2,2), is_subsample=True))
model.add(building_residual_block('res2', (1, 14, 14), 64, (3, 3), subsample=(2,2), is_subsample=True))
model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model_save_dir = '../saved_models/ResVGG_MNIST/'
model_save_filename = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'
# print(os.listdir(model_save_dir))
model.load_weights(model_save_dir + 'weights.14-0.99.hdf5')
# model.fit(x_train, y_train, batch_size=256, nb_epoch=15,
#           callbacks=[ModelCheckpoint(model_save_dir + model_save_filename)],
#           validation_split=0.2, show_accuracy=True)
print(model.evaluate(x_test, y_test, show_accuracy=True))