# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


def read_hyperparameters():
    filter_size = []
    filters_number = []

    layers = int(input("Give number of Convolution layers : "))

    print("Give number of convolutional filters every layer.One integer per line for every layer.")
    for i in range(layers):
        filters_number.append(int(input()))

    # print("Give filter size for every layer.Input format : dim1 dim2 per line for every layer")
    # for i in range(layers):
    #     a, b = map(int, input().split())
    #     filter_size.append((a,b))

    print("Give filter size.Input format : dim1 dim2 ")
    dim1, dim2 = map(int, input().split())
    filterSize = (dim1, dim2)

    epochs = int(input("Give number of epochs for training : "))

    batch_size = int(input("Give batch size : "))

    return layers, filters_number, filterSize, epochs, batch_size  # might need to change filterSize to filters_size


def encoder(image, filtersPerLayer, kernel_size):

    conv = image

    # conv = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))(image)

    counter = 0
    # encoder
    for layer_size in filtersPerLayer:
        conv = tf.keras.layers.Conv2D(filters=layer_size, kernel_size=kernel_size, activation='relu', padding='same',)(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        if counter < 2:
            conv = tf.keras.layers.MaxPooling2D(padding='same')(conv)
        counter += 1

    return conv


def decoder(conv, filtersPerLayer, kernel_size):
    counter = len(filtersPerLayer)
    # decoder
    for layer_size in reversed(filtersPerLayer):
        counter -= 1
        conv = tf.keras.layers.Conv2D(filters=layer_size, kernel_size=kernel_size, activation='relu', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        if counter < 2:
            conv = tf.keras.layers.UpSampling2D()(conv)

    image = tf.keras.layers.Conv2D(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same')(conv)

    return image

    # tf.keras.layers.Flatten(),
    # # No activation
    # tf.keras.layers.Dense(latent_dim + latent_dim)

def plot_error(autoencoder,  autoencoder_train, epochs, hyper_list, losses):
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'ro', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    for i in range(len(hyper_list[0])):
        toPlot = []
        for tup in hyper_list:
            toPlot.append(tup[i])
        plt.figure()
        plt.plot(toPlot, losses, 'b', label='Validation loss')
        plt.plot(toPlot, losses, 'ro', label='Validation loss')
        if i==0:
            plt.title('Convolution layers')
        elif i==1:
            plt.title('Filters at last convolution layer')
        elif i==2:
            plt.title('Kernel dimension')
        elif i==3:
            plt.title('Number of training epochs')
        elif i==4:
            plt.title('Batch size')
        plt.legend()
        plt.show()

    # conv_layers, filtersPerLayer, kernel_size, epochs, batch_size


if __name__ == '__main__':
    filename = 'train.dat'

    f = open(filename, "rb")

    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())

    f.close()

    images = np.array(pixels)
    images = images/np.max(images)
    images = images.reshape(-1, 28, 28, 1)

    action = 1
    hyper_list = []
    losses = []

    while action < 3:


        conv_layers, filtersPerLayer, kernel_size, epochs, batch_size = read_hyperparameters()
        hyper_list.append((conv_layers, filtersPerLayer[-1], kernel_size[0], epochs, batch_size))

        input_img = tf.keras.Input(shape=(28, 28, 1))

        # build the autoencoder model
        autoencoder = tf.keras.Model(input_img, decoder(encoder(input_img, filtersPerLayer, kernel_size), filtersPerLayer, kernel_size))
        autoencoder.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop())
        #autoencoder.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

        print(autoencoder.summary())

        train_X, valid_X, train_ground, valid_ground  = train_test_split(images, images, test_size=0.2, random_state=13)


        autoencoder_train = autoencoder.fit(train_X, train_X, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(valid_X, valid_X))
        #autoencoder_train contains info useful for plotting error

        losses.append(autoencoder_train.history['val_loss'][-1])

        #predict the validation set
        pred = autoencoder.predict(valid_X)

        print('Training complete. Choose an action.')
        print('1 - Repeat the training with different hyperparameters')
        print('2 - See plots relevant to this training procedure')
        print('3 - Save the current trained model')

        action = int(input())

        if action >= 3:
            break

        elif action == 2:
            plot_error(autoencoder, autoencoder_train, epochs, hyper_list, losses)

        print('Choose an action.')
        print('1,2 - Repeat the training with different hyperparameters')
        print('3 - Save the current trained model')

        action = int(input())


    autoencoder.save('autoencoder.h5')



    # filepath = './models/aa.h5'
    # tf.keras.models.save_model(autoencoder, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)
    #
    # tf.saved_model.save(autoencoder, export_dir, signatures=None, options=None)