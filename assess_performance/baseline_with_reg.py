import numpy as np
np.random.seed(1)

from source import DataNormalizer, create_mnist_model, FileRecord
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import pandas as pd
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(np.unique(y_test))
y_train_1dim = y_train
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
normalizer = DataNormalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)


# model variables
init_scheme = 'he_normal'
nb_epoch = 150

# filepaths
file_path = '../data/assess_baseline_with_reg.txt'

# cv
if '__main__' == __name__:
    input = open(file_path, 'a')
    input.write('loss,val_loss,acc,val_acc,itr\n')
    for train, test in StratifiedKFold(y_train_1dim, 10):
        x_train_sample, y_train_sample = x_train[train], y_train[train]
        x_test_sample, y_test_sample = x_train[test], y_train[test]

        # baseline model
        model = create_mnist_model(2, 50, l2(10**-4), 0, init_scheme)
        model.fit(x_train_sample, y_train_sample, batch_size=256, nb_epoch=nb_epoch,
                  verbose=1, show_accuracy=True, callbacks=[FileRecord(input)],
                  validation_data=(x_test_sample, y_test_sample))
        input.flush()

    input.close()


