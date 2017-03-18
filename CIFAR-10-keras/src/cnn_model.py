'''
'''
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import yaml
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle



class Model(object):

    def __init__(self):
        self.net = None
        self.build_net()
        self.compile_net()
        #
        self.onehot_encoder = OneHotEncoder(sparse=False)
        #
        self.config = None

    def parse_config(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.load(file)

    def onehot_encode(self, y_samples):
        y_samples = self.onehot_encoder.fit_transform(y_samples)
        return y_samples

    def build_net(self):
        self.net = Sequential()

        self.net.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
        self.net.add(Activation('relu'))
        self.net.add(Convolution2D(32, 3, 3))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        self.net.add(Dropout(0.25))

        self.net.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.net.add(Activation('relu'))
        self.net.add(Convolution2D(64, 3, 3))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        self.net.add(Dropout(0.25))

        self.net.add(Flatten())
        self.net.add(Dense(512))
        self.net.add(Activation('relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(10))
        self.net.add(Activation('softmax'))


        # self.net = Sequential()
        # self.net.add(Convolution2D(6, 5, 5, border_mode='same', input_shape=(3, 32, 32)))
        # self.net.add(Activation('relu'))
        # self.net.add(MaxPooling2D(pool_size=(2, 2)))
        # self.net.add(Dropout(0.25))
        #
        # self.net.add(Convolution2D(16, 5, 5))
        # self.net.add(Activation('relu'))
        # self.net.add(MaxPooling2D(pool_size=(2, 2)))
        # self.net.add(Dropout(0.25))
        #
        # # self.net.add(Convolution2D(64, 3, 3, border_mode='same'))
        # # self.net.add(Activation('relu'))
        # # self.net.add(Convolution2D(64, 3, 3))
        # # self.net.add(Activation('relu'))
        # # self.net.add(MaxPooling2D(pool_size=(2, 2)))
        # # self.net.add(Dropout(0.25))
        #
        # self.net.add(Flatten())
        # self.net.add(Dense(120))
        # self.net.add(Activation('relu'))
        # self.net.add(Dropout(0.5))
        #
        # self.net.add(Dense(10))
        # self.net.add(Activation('softmax'))

    def compile_net(self):

        self.net.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.net.summary()

    def train(self, x_train, y_train):
        y_train = y_train.reshape(-1, 1)
        y_train = self.onehot_encode(y_train)
        self.net.fit(x_train, y_train,
                     batch_size=self.config['batch_size'],
                     nb_epoch=self.config['n_epoch'],
                     validation_split=0.01,
                     shuffle=True)
        # Save trained model
        if self.config['save_trained_model'] is True:
            self.net.save(self.config['save_trained_model_path'])

    def predict(self, x_predict):
        if self.config['load_trained_model'] is True:
            self.net = keras.models.load_model(self.config['trained_model_path'])
            n_samples = len(x_predict)
            y_predict = self.net.predict_classes(x_predict)
            return y_predict
