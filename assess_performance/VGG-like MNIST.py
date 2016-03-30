from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from source import DataNormalizer, get_activations, mnist, FileRecord, CvIterator, CvTestPerformance
from model_repo import vgg_keras

# model_save_dir = '../saved_models/VGG_MNIST/'
# model_save_filename = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'
k = 10
nb_epoch = 100

(all_x_train, all_y_train), (real_x_test, real_y_test) = mnist(for_conv=True)
test = CvIterator(all_x_train, all_y_train, k=k, validation_split=0.2)
cv_recorder = CvTestPerformance('vggMnist')

i = 0
for ((x_train, y_train), (x_valid, y_valid)), (x_test, y_test) in test:
    model = vgg_keras()
    model_status_save = '../data/assess_vggMnist{0}.txt'.format(i)
    input = open(model_status_save, 'w')
    input.write('loss,val_loss,acc,val_acc,itr\n')
    file_recorder = FileRecord(input)
    callbacks = []
    callbacks.append(file_recorder)
    model.fit(x_train, y_train, batch_size=256, nb_epoch=nb_epoch,
              callbacks=callbacks, validation_data=(x_valid, y_valid), show_accuracy=True,
              verbose=2)
    cv_recorder.log(model.evaluate(x_test, y_test, show_accuracy=True))
    i += 1