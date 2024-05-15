import tensorflow as tf

import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_cifar():
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_fashionmnist():
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def load_medmnist():
    dataset= np.load('datasets/organamnist.npz')
    x_train=dataset['train_images']
    y_train= dataset['train_labels']
    x_test=dataset['test_images']
    y_test= dataset['test_labels']

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)

def load_chex():
    x_train= np.load('datasets/chexp_128_x_train.npy')
    x_test = np.load('datasets/chexp_128_x_test.npy')
    y_train = np.load('datasets/chexp_128_y_train.npy')
    y_test = np.load('datasets/chexp_128_y_test.npy')
    x_val = np.load('datasets/chexp_128_x_val.npy')
    y_val = np.load('datasets/chexp_128_y_val.npy')

    x_train = x_train.reshape(-1,128, 128, 1)
    x_test = x_test.reshape(-1, 128, 128, 1)
    x_val = x_val.reshape(-1, 128, 128, 1)
    #x_train_flat = x_train.reshape(-1, 16384)
    #x_test_flat = x_test.reshape(-1, 16384)
    #pca = PCA(n_components=784)
    #x_train_pca = pca.fit_transform(x_train_flat)
    #x_test_pca = pca.fit_transform(x_test_flat)
    #x_train = x_train_pca.reshape(-1, 28, 28,1)
    #x_test = x_test_pca.reshape(-1, 28, 28, 1)
    #y_train = to_categorical(y_train.astype('float32'))
    #y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)

def load_ret():

    x_train= np.load('datasets/4Xtrain4.npy')
    x_test = np.load('datasets/4Xtest4.npy')
    y_train = np.load('datasets/4ytrain4.npy')
    y_test = np.load('datasets/4ytest4.npy')

    x_train = x_train.reshape(-1, 128, 128, 3)
    x_test = x_test.reshape(-1, 128, 128, 3)
    #y_train = to_categorical(y_train.astype('float32'))
    #y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)

def load_blood():
    x_train= np.load('datasets/X_blooda_train1.npy')
    x_test = np.load('datasets/X_blooda_test1.npy')
    y_train = np.load('datasets/Y_blooda_train1.npy')
    y_test = np.load('datasets/Y_blooda_test1.npy')

    x_train = x_train.reshape(-1,128, 128, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 128, 128, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)