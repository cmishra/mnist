from keras.datasets import mnist
from keras.utils import np_utils
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from source import DataNormalizer, plot_performance


# load data, look at it
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(np.unique(y_test))
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# img = Image.fromarray(x_train[2], 'L')
# plt.imshow(x_train[2], cmap=cm.Greys_r)
# plt.show()

# create toy model
init_scheme = 'he_normal'
model = Sequential()
model.add(Flatten(input_shape=((28,28))))
model.add(Dense(50, input_dim=28*28, activation='relu', init=init_scheme))
model.add(Dense(50, activation='relu', init=init_scheme))
model.add(Dense(10, activation='softmax', init=init_scheme))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Model without normalization
num_epoches = 200
num_sample = 20
hist = model.fit(x_train[:num_sample], y_train[:num_sample],
                 batch_size=256,
                 nb_epoch=num_epoches,
                 show_accuracy=True,
                 verbose=0,
                 validation_split=0.2)

normalizer = DataNormalizer()
normalizer.fit(x_train[:num_sample])
x_train_transformed = normalizer.transform(x_train[:num_sample])


# Model with normalization
hist2 = model.fit(x_train_transformed, y_train[:num_sample],
                    batch_size=256, nb_epoch=num_epoches, verbose=1,
                 validation_split=0.2,
                 show_accuracy=True)
plt.subplot(211)
plot_performance(hist, hist2, 'acc', 'standard', 'normalized')
plt.subplot(212)
plot_performance(hist, hist2, 'loss', 'standard', 'normalized')
plt.show()

