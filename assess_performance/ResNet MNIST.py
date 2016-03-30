from keras.models import Sequential, Graph
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from source import mnist, graph_training_wrapper, CvIterator, CvTestPerformance, evaluate_softmax
from model_repo import res_vgg
from residual_blocks import building_residual_block

k = 10
nb_epoch = 100

# load data
(all_x_train, all_y_train), (real_x_test, real_y_test) = mnist(for_conv=True)
test = CvIterator(all_x_train, all_y_train, k=k, validation_split=0.2)
cv_recorder = CvTestPerformance('ResNet')
i = 0

for ((x_train, y_train), (x_valid, y_valid)), (x_test, y_test) in test:
    model = res_vgg()
    file_path = '../data/assess_ResNetFold{0!s}.txt'.format(i)
    graph_training_wrapper(model, x_train, y_train, x_valid, y_valid, nb_epoch,
                           training_progress_record=file_path)
    results = []
    results.append(model.evaluate({'input':x_test, 'output':y_test}))
    results.append(evaluate_softmax(model.predict({'input':x_test})['output'], y_test))
    cv_recorder.log(results)
    i += 1
