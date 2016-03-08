from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.regularizers import l2
from source import DataNormalizer, create_mnist_model


# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(np.unique(y_test))
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
normalizer = DataNormalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)


# # coarse hyperparameter tuner
# init_scheme = 'he_normal'
# num_runs = 100
# nb_epoch = 5
# num_layers = 2
# num_nodes = 50
# file = 'data/coarse_lambda_dropout_' + str(num_layers) + '_' + str(num_nodes) + '.txt'
# with open(file, 'a') as input:
#     input.write("training error" + ',' + "validation error" + ',' + "lambda" + ',' + 'dropout prob' + '\n')
# for i in range(num_runs):
#     l = 10.0**np.random.uniform(-5, 2)
#     w_regularizer = l2(l)
#     dropout_prob = np.random.uniform(0, 0.8)
#     model = create_mnist_model(num_layers, num_nodes, w_regularizer, dropout_prob, init_scheme)
#     hist = model.fit(x_train, y_train, batch_size=256, nb_epoch=nb_epoch,
#                     validation_split=0.2, verbose=0,
#                      show_accuracy=True)
#     with open(file, 'a') as input:
#         input.write(str(hist.history["val_acc"][-1]) + ',' +
#                       str(hist.history["acc"][-1]) + ',' +
#                       str(l) + ',' + str(dropout_prob) + '\n')
#     print(str(i) + '\t' + "training error: " + str(hist.history["val_acc"][-1]), "validation error: " + str(hist.history["acc"][-1]),
#           "lambda: " + str(l), 'dropout_prob: ' + str(dropout_prob), sep='\t')

# fine hyperparameter tuner
init_scheme = 'he_normal'
num_runs = 100
nb_epoch = 15
num_layers = 2
num_nodes = 50
file = '../data/fine_lambda_dropout_' + str(num_layers) + '_' + str(num_nodes) + '.txt'
with open(file, 'a') as input:
    input.write("training error" + ',' + "validation error" + ',' + "lambda" + ',' + 'dropout prob' + '\n')
for i in range(num_runs):
    l = 10.0**np.random.uniform(-5, -1)
    w_regularizer = l2(l)
    dropout_prob = np.random.uniform(0, 0.3)
    model = create_mnist_model(num_layers, num_nodes, w_regularizer, dropout_prob, init_scheme)
    hist = model.fit(x_train, y_train, batch_size=256, nb_epoch=nb_epoch,
                    validation_split=0.2, verbose=0,
                     show_accuracy=True)
    with open(file, 'a') as input:
        input.write(str(hist.history["val_acc"][-1]) + ',' +
                      str(hist.history["acc"][-1]) + ',' +
                      str(l) + ',' + str(dropout_prob) + '\n')
    print(str(i) + '\t' + "training error: " + str(hist.history["val_acc"][-1]), "validation error: " + str(hist.history["acc"][-1]),
          "lambda: " + str(l), 'dropout_prob: ' + str(dropout_prob), sep='\t')
