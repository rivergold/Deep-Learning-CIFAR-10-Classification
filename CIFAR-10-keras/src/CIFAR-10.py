'''
This scrpt trains a CNN model for image classifiction using PtTorch
'''

import numpy as np
import argparse
import pickle
from cnn_model import *

from sklearn.metrics import classification_report, confusion_matrix

class CIFAR_10(object):
    def __init__(self):
        self.cnn_model = Model()
        self.label_names = None

    def parse_args(self):
        argparser = argparse.ArgumentParser()
        argparser.add_argument('-m', '--mode', choices=['train', 'test', 'predict'])
        argparser.add_argument('-p', '--path')
        return argparser.parse_args()

    def load_samples(self, file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)

    def load_label_names(self, file_name):
        with open(file_name, 'rb') as file:
            self.label_names = pickle.load(file)

    def train(self, x_train, y_train):
        self.cnn_model.train(x_train, y_train)

    def predict(self, x_predict):
        y_predict = self.cnn_model.predict(x_predict)
        return y_predict

    def test(self, x_test, y_test):
        y_predict = self.predict(x_test)
        # =====================
        # debug
        # print('y_predict is {}'.format(y_predict))
        # print('y_true is {}'.format(y_test))
        # =====================
        print(classification_report(y_test, y_predict, target_names=self.label_names))
        print(confusion_matrix(y_test, y_predict, labels=range(10)))
        return y_predict

    def run(self):
        args = self.parse_args()
        # Load config file.
        self.cnn_model.parse_config(args.path + '/config.yaml')
        # Load label names.
        self.load_label_names(args.path + '/CIFAR_10-Label-Names.pkl')
        #
        if args.mode == 'train':
            print('-> Loading train samples...')
            samples = self.load_samples(args.path + '/CIFAR-10-Train.pkl')
            x_train = np.array(samples['data']).astype(np.float32) / 255
            y_train = np.array(samples['label']).astype(np.int64)
            print('-> Finished loading {} samples.'.format(len(y_train)))
            # Trainning
            print('-> Training')
            self.train(x_train, y_train)

        elif args.mode == 'test':
            print('-> Loading test samples...')
            samples = self.load_samples(args.path + '/CIFAR-10-Test.pkl')
            x_test = np.array(samples['data']).astype(np.float32) / 255
            y_test = np.array(samples['label']).astype(np.int64)
            print('-> Finished loading {} samples.'.format(len(y_test)))
            #
            self.test(x_test, y_test)

        elif args.mode == 'predict':
            pass


if __name__ == '__main__':
    cifar_10 = CIFAR_10()
    cifar_10.run()
    # print(cifar_10.y_train)
