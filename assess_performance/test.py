from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator


nb_epoch = 10
(all_x_train, all_y_train), ignore = mnist.load_data()
all_y_train = np_utils.to_categorical(all_y_train)

generator = ImageDataGenerator()
generator.fit(all_x_train)
for x_batch, y_batch in generator.flow(all_x_train, all_y_train, batch_size=100):
    x_train = x_batch.reshape((100, 1, 28, 28))
    y_train = y_batch
    break

print("~~~~~~~~~~~~~~ Control Border")
model1 = Sequential()
model1.add(Convolution2D(32, 5, 5, subsample=(2,2), input_shape=(1, 28, 28), activation='relu'))
model1.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
model1.add(Convolution2D(128, 3, 3, subsample=(2,2), activation='relu'))
model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam')
model1.fit(x_train, y_train, show_accuracy=True, nb_epoch=nb_epoch)

print("~~~~~~~~~~~~~~ Control Stride")
model1 = Sequential()
model1.add(Convolution2D(32, 5, 5, subsample=(2,2), input_shape=(1, 28, 28), activation='relu'))
model1.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
model1.add(Convolution2D(128, 3, 3, subsample=(1,1), activation='relu'))
model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam')
model1.fit(x_train, y_train, show_accuracy=True, nb_epoch=nb_epoch)

print("~~~~~~~~~~~~~~ Test")
model2 = Sequential()
model2.add(Convolution2D(32, 5, 5, subsample=(2,2), input_shape=(1, 28, 28), activation='relu', border_mode='same'))
model2.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu', border_mode='same'))
model2.add(Convolution2D(128, 3, 3, subsample=(2,2), activation='relu', border_mode='same'))
model2.add(Flatten())
model2.add(Dense(10, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam')
model2.fit(x_train, y_train, show_accuracy=True, nb_epoch=nb_epoch)
