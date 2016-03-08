from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.layers import Layer, Dense
from keras.models import Graph, Sequential
import numpy as np
from source import DataNormalizer, get_activations

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(np.unique(y_test))
y_train_1dim = y_train
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
normalizer = DataNormalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)

model = Graph()
model.add_input(name='input', input_shape=(28,28))
model.add_node(Flatten(), name='flatten', input='input')

test_submission = Sequential()
test_submission.add(Dense(20, activation='relu', input_dim=28*28))
test_submission.add(Dense(20, activation='relu'))
test_submission.add(Dense(20, activation='relu'))

model.add_node(test_submission, name='path', input='flatten')
model.add_node(Dense(20, activation='relu'),
          name='path2', input='flatten')
model.add_node(Dense(10, activation='softmax'),
          name='recombine', inputs=['path', 'path2'],
          merge_mode='sum')
model.add_output(name='output', input='recombine')

model.compile(optimizer='rmsprop', loss={'output':'categorical_crossentropy'})

hist = model.fit({'input': x_train, 'output':y_train}, nb_epoch=1,
          validation_split=0.2)

for l in test_submission.layers:
    print(str(l.input_shape) + ' ' + str(l.output_shape))