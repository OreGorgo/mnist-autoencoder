# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D



class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [

                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


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

    return layers, filters_number, filterSize  # might need to change filterSize to filters_size

#AYTO EDW MPOROYME NA TO XRHSIMOPOIHSOUME
def build_model(hidden_layer_sizes):
  model = Sequential()

  model.add(Dense(hidden_layer_sizes[0], input_dim=2))
  model.add(Activation('tanh'))

  for layer_size in hidden_layer_sizes[1:]:
    model.add(Dense(layer_size))
    model.add(Activation('tanh'))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  return model



if __name__ == '__main__':
    filename = 'test.dat'

    f = open(filename, "rb")

    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())

    f.close()

    print(magic)
    print(size)
    print(rows)
    print(cols)

    images = np.array(pixels)
    images = images.reshape([-1, 28, 28, 1])

    print(images.shape)
    #
    # encoder=Sequential()
    # x = Convolution2D(128, 3, 3, activation='relu', padding='same')(images[0])
    # x = MaxPooling2D((2, 2), border_mode='same')(x)

    # plt.imshow(images[899], cmap=plt.cm.binary)
    # plt.show()

    # auto = CVAE(10)
    #
    # print(images[0].shape)
    #
    # a = auto.encode(images)
    #
    # print(a[0].shape)
    # print(a[1].shape)

    conv_layers, filtersPerLayer, filter_size = read_hyperparameters()

    print(conv_layers,filtersPerLayer[0],filter_size)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
