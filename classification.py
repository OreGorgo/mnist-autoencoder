import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import sys
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def read_hyperparameters():

    nodes = int(input("Give number of nodes in fully connected layer : "))

    epochs = int(input("Give number of epochs for training : "))

    batch_size = int(input("Give batch size : "))

    return nodes, epochs, batch_size  # might need to change filterSize to filters_size


def read_data(train_file, train_file_labels, test_file, test_file_labels):

    #read train images
    f = open(train_file, "rb")

    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())
    f.close()

    images = np.array(pixels)
    images = images / np.max(images)
    train_images = images.reshape(-1, 28, 28, 1)

    # read test images
    f = open(test_file, "rb")
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())
    f.close()

    images = np.array(pixels)
    images = images / np.max(images)
    test_images = images.reshape(-1, 28, 28, 1)

    # read teain labels
    f = open(train_file_labels, "rb")
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    labels = list(f.read())

    f.close()
    train_labels = np.array(labels)

    # read test labels
    f = open(test_file_labels, "rb")
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    labels = list(f.read())

    f.close()
    test_labels = np.array(labels)

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':

    #this should be command line input
    train_file = 'train.dat'
    train_file_labels = 'train_labels.dat'
    test_file = 'test.dat'
    test_file_labels = 'test_labels.dat'
    model_file = 'autoencoder.h5'

    #read input from command line
    for it, arg in enumerate(sys.argv[1:]):
        if arg=='-d':
            train_file = sys.argv[it+1]
        elif arg == '-dl':
            train_file_labels = sys.argv[it + 1]
        elif arg == '-t':
            test_file = sys.argv[it + 1]
        elif arg == '-tl':
            test_file_labels = sys.argv[it + 1]
        elif arg == '-model':
            model_file = sys.argv[it + 1]


    #read data from files
    train_images, train_labels, test_images, test_labels = read_data(train_file, train_file_labels, test_file, test_file_labels)

    print(train_labels)

    autoencoder = load_model(model_file)
    encoder_length=len(autoencoder.layers)//2

    action = 1

    while action < 3:
        model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[encoder_length-1].output)

        nodes, epochs, batch_size = read_hyperparameters()

        classifier = tf.keras.Sequential()
        classifier.add(model)

        classifier.add(tf.keras.layers.Flatten())
        classifier.add(tf.keras.layers.Dense(nodes, activation='relu'))
        classifier.add(tf.keras.layers.Dense(10, activation='softmax'))


        #classifier.add(tf.keras.layers.Softmax())

        classifier.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=['accuracy'])

        classifier.summary()

        classifier.layers[0].trainable = False
        classifier.layers[1].trainable = False
        classifier.layers[3].trainable = False

        classifier.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=1, use_multiprocessing=True)


        print('Training complete. Choose an action.')
        print('1 - Repeat the training with different hyperparameters')
        print('2 - See plots relevant to experiments up til now')
        print('3 - Use current classifier model on test set')

        action = int(input())

        if action >= 3:
            break

        #elif action == 2:
            #plots(autoencoder, autoencoder_train, epochs)

        print('Choose an action.')
        print('1,2 - Repeat the training with different hyperparameters')
        print('3 - Save the current trained model')

    pred = classifier.predict(test_images, verbose=1, use_multiprocessing=True)

    y_pred = np.argmax(pred, axis=1)

    print(pred)
    print(y_pred)


    # Print f1, precision, and recall scores
    print(precision_score(test_labels, y_pred, average="macro"))
    print(recall_score(test_labels, y_pred, average="macro"))
    print(f1_score(test_labels, y_pred, average="macro"))



    # loss, accuracy, f1_score, precision, recall = classifier.evaluate(test_images, test_labels, batch_size=batch_size)
    #
    # print("test loss", loss)
    # print("test accuracy", accuracy)
    # print("test f1_score", f1_score)
    # print("test precision", precision)
    # print("test recall", recall)

    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


    # for layer in model.layers:
    #     if(layer.get_weights()):
    #         print(layer.get_config(), layer.get_weights()[0], layer.output_shape, layer.input_shape)
    #
    # for layer in autoencoder.layers:
    #     if(layer.get_weights()):
    #         print(layer.get_config(), layer.get_weights()[0], layer.output_shape, layer.input_shape)
