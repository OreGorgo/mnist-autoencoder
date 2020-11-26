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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def read_hyperparameters():
    #reads all the hyperparameters for one experiment

    nodes = int(input("Give number of nodes in fully connected layer : "))
    epochs = int(input("Give number of epochs for training : "))
    batch_size = int(input("Give batch size : "))

    return nodes, epochs, batch_size


def read_data(train_file, train_file_labels, test_file, test_file_labels):

    #read train images
    f = open(train_file, "rb")

    #unpack data
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())
    f.close()

    #reshape and normalize pixels
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

def plots(autoencoder,  autoencoder_train, epochs, hyper_list, losses, X_train, y_train):
    #plot loss vs epochs for both train and validation sets
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'ro', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    #plot accuracy vs epochs for both train and validation sets
    accuracy = autoencoder_train.history['accuracy']
    val_accuracy = autoencoder_train.history['val_accuracy']
    plt.figure()
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'ro', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = [i for i in range(0, 10)]
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx][:, :, 0])
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

    #plot loss vs hyperparameters for all past experiments
    for i in range(len(hyper_list[0])):
        toPlot = []
        for tup in hyper_list:
            toPlot.append(tup[i])
        plt.figure()
        plt.plot(toPlot, losses, 'b', label='Validation loss')
        plt.plot(toPlot, losses, 'ro', label='Validation loss')
        if i == 0:
            plt.title('Fully connected nodes')
        elif i == 1:
            plt.title('Number of training epochs')
        elif i == 2:
            plt.title('Batch size')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    #Fallback values for the command line inputs
    train_file = 'train.dat'
    train_file_labels = 'train_labels.dat'
    test_file = 'test.dat'
    test_file_labels = 'test_labels.dat'
    model_file = 'good_autoencoder.h5'

    #read input from command line
    for it, arg in enumerate(sys.argv):
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

    # load autoencoder
    autoencoder = load_model(model_file)
    encoder_length=len(autoencoder.layers)//2

    action = 1
    losses=[]
    hyper_list=[]

    while action < 3:

        print('Do you want to load a pre-trained classifier? Type yes (y) or no (n)')
        answer = input()

        if answer == 'y' or answer == 'yes':
            classifier_path = input("Type the pre-trained classifier path. ")
            classifier = load_model(classifier_path)
            break

        #create a new model from the encoder part of autoencoder
        model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[encoder_length-1].output)

        nodes, epochs, batch_size = read_hyperparameters()
        hyper_list.append((nodes, epochs, batch_size))

        # Initialize the classifier and add the encoder layers to it
        classifier = tf.keras.Sequential()
        classifier.add(model)

        #add flatten, fully connected and output layers
        classifier.add(tf.keras.layers.Flatten())
        classifier.add(tf.keras.layers.Dense(nodes, activation='relu'))
        #classifier.add(tf.keras.layers.Dropout(0.1))
        classifier.add(tf.keras.layers.Dense(10, activation='softmax'))


        classifier.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

        classifier.summary()

        #make encoder non trainable
        classifier.layers[0].trainable = False
        classifier.layers[1].trainable = False
        classifier.layers[3].trainable = False

        #train the fully connected layer only
        classifier_train = classifier.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=1, use_multiprocessing=True)

        losses.append(classifier_train.history['loss'][-1])

        #train the entire model
        classifier.layers[0].trainable = True
        classifier.layers[1].trainable = True
        classifier.layers[3].trainable = True

        classifier.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=1, use_multiprocessing=True)


        print('Training complete. Choose an action.')
        print('1 - Repeat the training with different hyperparameters')
        print('2 - See plots relevant to experiments up til now')
        print('3 - Use current classifier model on test set')

        action = int(input())

        if action >= 3:
            break

        elif action == 2:
            plots(classifier, classifier_train, epochs, hyper_list, losses, train_images, train_labels)

        print('Choose an action.')
        print('1,2 - Repeat the training with different hyperparameters')
        print('3 - Save the current trained model')

        action = int(input())

    pred = classifier.predict(test_images, verbose=1, use_multiprocessing=True)
    y_pred = np.argmax(pred, axis=1)


    # Print f1, precision, and recall scores

    print("Accuracy of test set:")
    print(accuracy_score(test_labels, y_pred))


    print('Classification Report:')
    print(classification_report(test_labels, y_pred))


    classifier.save('classifier.h5')
