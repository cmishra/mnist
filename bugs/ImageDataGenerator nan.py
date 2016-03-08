from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

sample_size = 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_sample, y_sample = x_train[:sample_size], y_train[:sample_size]
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
datagen.fit(x_sample)
i = 0
for x, y in datagen.flow(x_sample, y_sample, batch_size=256):
    print("original: --------------")
    print(x_sample[i])
    print("transformed: -----------")
    print(x)
    i += 1
    if i == sample_size:
        break
