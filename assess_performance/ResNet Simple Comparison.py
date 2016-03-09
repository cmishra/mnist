import numpy as np
np.random.seed(1)
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.layers import Layer, Dense
from keras.models import Graph, Sequential
from keras.optimizers import SGD
import os
from source import DataNormalizer, get_activations, Identity, FileRecord, evaluate_softmax

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
model.add_node(Dense(50, activation='relu'), input='flatten',
               name='50_dims')
test_submission = Sequential()
test_submission.add(Dense(50, activation='relu', input_dim=50))
test_submission.add(Dense(50, activation='relu'))
test_submission.add(Dense(50))
model.add_node(test_submission, name='node3', input='50_dims')
model.add_node(Activation('relu'), input='node3',
               name='recombine')
model.add_node(Dense(10, activation='softmax'),
          name='prediction_layer', input='recombine')
model.add_output(name='output', input='prediction_layer')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss={'output':'categorical_crossentropy'})

model_save_dir = '../saved_models/ResNet_Comparison/'
model_save_filename = 'weights.{val_loss:.2f}.hdf5'

shuffled_indexes = np.arange(x_train.shape[0])
np.random.shuffle(shuffled_indexes)
x_train = x_train[shuffled_indexes]
y_train = y_train[shuffled_indexes]
x_valid = x_train[:12000]
y_valid = y_train[:12000]
x_train = x_train[12000:]
y_train = y_train[12000:]

file_path = '../data/assess_resnetCompare.txt'

input = open(file_path, 'a')
input.write('loss,val_loss,acc,val_acc,itr\n')
file_recorder = FileRecord(input)
model_saver = ModelCheckpoint(model_save_dir + model_save_filename)
for i in range(20):
    model.fit({'input': x_train, 'output':y_train}, nb_epoch=1,
          validation_data={'input': x_valid, 'output':y_valid}, verbose=0,
                 callbacks=[model_saver, file_recorder]
              )
    acc = evaluate_softmax(model.predict({'input':x_train})['output'], y_train)
    val_acc = evaluate_softmax(model.predict({'input':x_valid})['output'], y_valid)
    file_recorder.set_acc(acc, val_acc)
    print(file_recorder.counter - 1, 'Error on Train:', acc,
        '\tError on Validation:', val_acc, sep='\t')



input.close()

