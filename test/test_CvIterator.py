from source import CvIterator
import numpy as np


x_train = np.arange(10)
y_train = np.arange(10)

k = 3
test = CvIterator(x_train, y_train, k)
print(test.fold_indexes)

for ((x_train, y_train), (x_valid, y_valid)), (x_test, y_test) in test:
    print("test: " + str((x_test, y_test)))
    print("train: " + str(((x_train, y_train), (x_valid), (y_valid))))
