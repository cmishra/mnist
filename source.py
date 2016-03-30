import numpy as np, pandas as pd, math
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential, Graph
from keras.layers import Dense, Flatten, Dropout, Layer
from theano import function, config
from keras.utils import np_utils
from keras.datasets.mnist import load_data
import ggplot as gg

# normalizes images featurewise
class DataNormalizer:
    
    def fit(self, x):
        self.mean = np.mean(x, 0)
        self.std = np.std(x, 0)

    def transform(self, x):
        x = x - self.mean
        x = x / self.std
        nans = np.isnan(x)
        x[nans] = 0.0
        return x


class CvIterator():
    def __init__(self, x_train, y_train, k, validation_split=0.2):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.fold_indexes = []
        self.validation_split = validation_split
        self.shuffled_indexes = np.arange(x_train.shape[0])
        np.random.shuffle(self.shuffled_indexes)

        items_per_fold = math.floor(x_train.shape[0]/k)
        for i in range(k):
            self.fold_indexes.append(self.shuffled_indexes[i*items_per_fold:(i+1)*items_per_fold])

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.k:
            raise StopIteration
        else:
            test_indexes = self.fold_indexes[self.i]
            train_indexes = np.setdiff1d(self.shuffled_indexes, test_indexes)
            x_test = self.x_train[test_indexes]
            y_test = self.y_train[test_indexes]
            x_train = self.x_train[train_indexes]
            y_train = self.y_train[train_indexes]
            self.i += 1
            return set_validation(x_train, y_train, self.validation_split), (x_test, y_test)


# plot histories
def plot_performance(hist1, hist2, metric, label1='hist1', label2='hist2'):
    from matplotlib import pyplot as plt
    val_key, key = 'val_' + metric, metric 
    num_epoches = len(hist1.history[metric])
    plt.plot(np.arange(num_epoches), hist1.history[key], label=label1, color='orange')
    plt.plot(np.arange(num_epoches), hist1.history[val_key], label=label1 + '_val', color='red')
    if hist2 != None:
        plt.plot(np.arange(num_epoches), hist2.history[key], label=label2, color='aqua')
        plt.plot(np.arange(num_epoches), hist2.history[val_key], label=label2 + '_val', color='blue')
    plt.legend()
    plt.title(metric)


def create_mnist_model(num_layers, n_hidden, w_regularizer, dropout_prob, init_scheme):
    model = Sequential()
    model.add(Flatten(input_shape=((28,28))))
    for i in range(num_layers):
        model.add(Dense(n_hidden, input_dim=28*28, activation='relu', init=init_scheme,
                        W_regularizer=w_regularizer))
        model.add(Dropout(dropout_prob))
    model.add(Dense(10, activation='softmax', init=init_scheme,
                    W_regularizer=w_regularizer))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def evaluate_softmax(y_pred, y_actual):
    max = np.amax(y_pred, 1)
    is_max = y_pred == max.reshape((len(y_pred),1))
    y_pred[np.invert(is_max)] = 0
    y_pred[is_max] = 1
    num_wrong = np.abs(y_pred - y_actual).sum()/2
    return 1 - num_wrong/y_actual.shape[0]


class CvTestPerformance():
    def __init__(self, unique_name, file='../data/CV_Output.csv'):
        self.input = open(file, 'a')
        self.name = unique_name
        self.i = 0

    def log(self, results):
        assert len(results) == 2
        self.input.write(self.name + str(self.i) +',' + str(results[0]) + ',' + str(results[1]) + '\n')
        self.i += 1
        self.input.flush()


class FileRecord(Callback):

    def __init__(self, output_file):
        self.output = output_file
        self.counter = 1
        self.tmp_str = ''

    def on_epoch_end(self, epoch, logs={}):
        self.output.write(self.tmp_str)
        line = ''
        num_empty = 0
        for l in ['loss', 'val_loss', 'acc', 'val_acc']:
            if l in logs:
                line += str(logs.get(l)) + ','
            else:
                line += '{' + str(num_empty) + '},'
                num_empty += 1
        line += str(self.counter) + '\n'
        self.tmp_str = line
        self.counter += 1

    def set_acc(self, acc='', val_acc=''):
        self.tmp_str = self.tmp_str.format(acc, val_acc)

    def __del__(self):
        self.output.write(self.tmp_str)
        self.output.close()


def print_full_pd(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')


    
# Code from a keras github issue on how to do this
def get_activations(model, layer, X_batch):
    get_activations = function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations


class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)


def matrix_to_df(matrix, names):
    assert(isinstance(matrix, np.ndarray))
    assert(len(matrix.shape) == 2)
    cols = {}
    for i in range(matrix.shape[1]):
        cols[names[i]] = matrix[:,i]
    return pd.DataFrame(cols)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))


def gen_sequences(arr, seq_len, predictand=True):
    n = len(arr)
    if n % seq_len == 0:
        nsamples = n // seq_len - 1
    else:
        nsamples = n // seq_len
    seq = np.empty((nsamples, seq_len, 1), dtype='float32')
    shift = 1 if predictand else 0
    for i, c in enumerate(range(0, nsamples*seq_len, seq_len)):
        seq[i] = arr[c+shift:c+seq_len+shift].reshape((seq_len, 1))
    return seq


def collapse_sequences(sequences):
    tot_seq_count = sum(seq.shape[0] for seq in sequences)
    seq_len = sequences[0].shape[1]
    master_seq = np.empty((tot_seq_count, seq_len, 1))

    i = 0
    for seq in sequences:
        master_seq[i:i+seq.shape[0]] = seq
        i += seq.shape[0]
    return master_seq


# Accepts list of sequences which you want to split across test/train based on rows
# Assumes lengths of all sequences passed in are equivalent
def test_train_split(sequences, testing_prop=0.3, seed=None):
    n = sequences[0].shape[0]
    if seed is not None:
        np.random.seed(seed)
    training_indexes = np.random.randint(0, n, n - int(n*testing_prop))
    testing_indexes = np.setdiff1d(np.arange(n), training_indexes)
    training = list(seq[training_indexes] for seq in sequences)
    testing = list(seq[testing_indexes] for seq in sequences)
    return training, testing


class NormalizeTS(object):
    def __init__(self):
        self.dict = {}

    # Here MAD is defined as mean absolute deviation about the median
    def set_linear_transformation(self, seq, key):
        med_key, mad_key = NormalizeTS.keys_string(key)
        seq = seq.flatten()
        self.dict[med_key] = np.median(seq)
        self.dict[mad_key] = abs(seq - self.dict[med_key]).mean()

    @staticmethod
    def keys_string(key):
        return key + "_median", key + "_mad"

    def normalize(self, seq, key):
        med_key, mad_key = NormalizeTS.keys_string(key)
        return (seq - self.dict[med_key]) / self.dict[mad_key]

    def restore(self, seq, key):
        med_key, mad_key = NormalizeTS.keys_string(key)
        return seq * self.dict[mad_key] + self.dict[med_key]

if __name__ == '__main__':
    arr = np.array(list(range(11)))
    arr2 = np.array(list(range(20, 31)))
    x_1 = gen_sequences(arr, 2, False)
    x_2 = gen_sequences(arr2, 2, False)
    y_1 = gen_sequences(arr, 2)
    y_2 = gen_sequences(arr2, 2)
    x = collapse_sequences([x_1, x_2])
    y = collapse_sequences([y_1, y_2])

    preprocessor = NormalizeTS()
    preprocessor.set_linear_transformation(x, "x")
    preprocessor.set_linear_transformation(y, "y")
    x_norm = preprocessor.normalize(x, "x")
    y_norm = preprocessor.normalize(y, "y")
    x_unnorm = preprocessor.restore(x_norm, "x")
    y_unnorm = preprocessor.restore(y_norm, "y")

    sequences = [x, y]
    (x_train, y_train), (x_test, y_test) = test_train_split(sequences, 0.25)


def mnist(for_conv=False):
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    num_classes = len(np.unique(y_test))
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    normalizer = DataNormalizer()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    return (x_train, y_train), (x_test, y_test)


def set_validation(x_train, y_train, validation_split=0.2):
    size_of_valid = validation_split*x_train.shape[0]
    shuffled_indexes = np.arange(x_train.shape[0])
    np.random.shuffle(shuffled_indexes)
    x_train = x_train[shuffled_indexes]
    y_train = y_train[shuffled_indexes]
    x_valid = x_train[:size_of_valid]
    y_valid = y_train[:size_of_valid]
    x_train = x_train[size_of_valid:]
    y_train = y_train[size_of_valid:]
    return (x_train, y_train), (x_valid, y_valid)


def graph_training_wrapper(model, x_train, y_train, x_valid, y_valid, nb_epoch=1000, \
                           model_save_dir=None, model_save_filename=None, training_progress_record=None):
    callbacks = []
    if training_progress_record:
        input = open(training_progress_record, 'w')
        input.write('loss,val_loss,acc,val_acc,itr\n')
        file_recorder = FileRecord(input)
        callbacks.append(file_recorder)
    if model_save_filename and model_save_dir:
        callbacks.append(ModelCheckpoint(model_save_dir + model_save_filename))
    for i in range(nb_epoch):
        model.fit({'input': x_train, 'output':y_train}, nb_epoch=1, verbose=0,
              validation_data={'input': x_valid, 'output':y_valid}, callbacks=callbacks)
        acc = evaluate_softmax(model.predict({'input':x_train})['output'], y_train)
        val_acc = evaluate_softmax(model.predict({'input':x_valid})['output'], y_valid)
        if training_progress_record:
            file_recorder.set_acc(acc, val_acc)
            print(i, 'Accuracy on Train:', acc,
                'Accuracy on Validation:', val_acc, sep='\t')
        else:
            print(i, 'Accuracy on Train:', acc,
                'Accuracy on Validation:', val_acc, sep='\t')