'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def mnist_model(num_classes, img_rows, img_cols):
    """
    Returns keras model initialized for use on mnist data set
    :param num_classes: int
    :param img_rows: int
    :param img_cols: int
    :return: keras model
    """
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def make_blank_images(n_images, mean, std, shape):
    out_img = std * np.random.randn(n_images, *shape) + mean
    # make sure between -1 and 1
    out_img[out_img > 1] = 1
    out_img[out_img < -1] = -1
    return out_img


def process_data(x_train, x_test, y_train, y_test, num_classes=10):
    """
    Make data ready for training
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    img_rows, img_cols = x_train.shape[1:]
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test


def single_image(x):
    img_rows, img_cols = x.shape
    if K.image_data_format() == 'channels_first':
        x_out = x.reshape(1, 1, img_rows, img_cols)
    else:
        x_out = x.reshape(1, img_rows, img_cols, 1)

    x_out = x_out.astype('float32')
    x_out /= 255
    return x_out

def add_noise():
    null_class_train = int(x_train.shape[0] * .1)
    empty = np.stack([255 * ocr_img_gen.speckle((28, 28), 1) for i in range(null_class_train)])
    x_train = np.vstack((x_train, empty))
    y_train = np.concatenate((y_train, 10 * np.ones(null_class_train)))


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
