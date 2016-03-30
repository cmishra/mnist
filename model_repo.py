from keras.models import Sequential, Graph
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from math import floor
from source import get_activations
import numpy as np


def vgg_keras():
    # http://keras.io/examples/#vgg-like-convnet
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model


def plain_vgg(num_layers=6, border_mode='same', subsample=(2,2), input_shape=(1,28,28)):
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, subsample=(2,2), border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    num_layers -= 2 # for above layer and for FC output layer
    for i in range(floor(num_layers/2)):
        model.add(Convolution2D(32, 3, 3, border_mode=border_mode))
        model.add(Activation('relu'))
    for i in range(floor(num_layers/2), floor(2*num_layers/2)):
        if i == floor(num_layers/2):
            model.add(Convolution2D(64, 3, 3, subsample=subsample, border_mode=border_mode))
            model.add(Activation('relu'))
        else:
            model.add(Convolution2D(64, 3, 3, border_mode=border_mode))
            model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def res_vgg():
    model = Graph()
    model.add_input('input', (1,28,28))

    # First Residual Block Main Path
    res_path1 = Sequential()
    res_path1.add(Convolution2D(32, 3, 3, input_shape=(1, 28, 28), border_mode='same',
                                activation='relu'))
    res_path1.add(Convolution2D(32, 3, 3, border_mode='same'))

    # First Residual Block Shortcut
    shortcut1 = Sequential()
    shortcut1.add(Convolution2D(32, 1, 1, border_mode='same', input_shape=(1, 28, 28)))

    # Add first Residual Block
    model.add_node(res_path1, name='res_path1', input='input')
    model.add_node(shortcut1, input='input', name='shortcut_1')
    model.add_node(Activation('relu'), inputs=['shortcut_1', 'res_path1'],
                   name='res_output1', merge_mode='sum')

    # Second residual block main path
    res_path2 = Sequential()
    res_path2.add(MaxPooling2D(pool_size=(2, 2), input_shape=(32, 28, 28)))
    res_path2.add(Dropout(0.25))
    res_path2.add(Convolution2D(64, 3, 3,
                                activation='relu', border_mode='same'))
    res_path2.add(Convolution2D(64, 3, 3, border_mode='same'))

    #Second residual block shortcut
    shortcut2 = Sequential()
    shortcut2.add(Convolution2D(64, 1, 1, input_shape=(32, 28, 28),
                                 border_mode='same'))
    shortcut2.add(MaxPooling2D(pool_size=(2, 2)))

    # Add second residual block
    model.add_node(res_path2, name='res_path2', input='res_output1')
    model.add_node(shortcut2, name='shortcut_2', input='res_output1')
    model.add_node(Activation('relu'), inputs=['res_path2', 'shortcut_2'],
                   name='res_output2', merge_mode='sum')

    # Mapping to output
    output_mapping = Sequential()
    output_mapping.add(MaxPooling2D(pool_size=(2, 2), input_shape=(64, 14, 14)))
    output_mapping.add(Dropout(0.25))
    output_mapping.add(Flatten())
    output_mapping.add(Dense(256, activation='relu'))
    output_mapping.add(Dropout(0.5))
    output_mapping.add(Dense(10, activation='softmax'))

    # Add output_mapping to graph
    model.add_node(output_mapping, name='convnets_to_output', input='res_output2')
    model.add_output('output', input='convnets_to_output')

    model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam')
    return model


def conv_test():
    model = Sequential()
    model.add(ZeroPadding2D(padding=(1,1), input_shape=(1, 28, 28)))
    model.add(Convolution2D(32, 3, 3, subsample=(2,2), border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    for layer in range(len(model.layers)):
        print(get_activations(model, layer, np.ones((32, 1,28, 28))).shape)


