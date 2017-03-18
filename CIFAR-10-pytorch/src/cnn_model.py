'''
'''
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import yaml
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model(object):

    def __init__(self):
        # use a Classification Cross-Entropy loss
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        # Config
        self.config = None
        # Num of batch
        self.n_batch = None
        # Batch size
        self.batch_size = None
        # Samples id
        self.samples_id = None

    def parse_config(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.load(file)

    def preprocess_train_samples(self, x_train, y_train):
        n_samples = len(x_train)
        self.samples_id = range(n_samples)
        # Shuffle train data
        if self.config['shuffle train data'] is True:
            self.samples_id = shuffle(self.samples_id)
        # Divide batch
        self.batch_size = int(self.config['batch_size'])
        self.n_batch = math.ceil(n_samples / self.batch_size)

    def train(self, x_train, y_train):
        for epoch in range(int(self.config['n_epoch'])):
            running_loss = 0.0
            # Train each batch
            for batch_id in range(self.n_batch):
                # Batch samples id.
                batch_samples_id = self.samples_id[
                    batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
                # Get the inputs.
                inputs = torch.from_numpy(x_train[batch_samples_id])
                labels = torch.from_numpy(y_train[batch_samples_id])
                # Wrap them in Variable.
                inputs, labels = Variable(inputs), Variable(labels)
                # Zero the parameter gradients.
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # Update statistics
                running_loss += loss.data[0]
            # Print
            print('\nepoch={}, loss={:.10f}\n'.format(
                epoch + 1, running_loss / (batch_id * len(x_train))))
            running_loss = 0.0
        print('Finished Training')
        if self.config['save_trained_model'] is True:
            print('Saving trained model')
            torch.save(self.net.state_dict(), self.config[
                       'save_trained_model_path'])
            print('Finished saving')

    def predict(self, x_predict):
        if self.config['load_trained_model'] is True:
            trained_cnn_net = Net()
            trained_cnn_net.load_state_dict(
                torch.load(self.config['trained_model_path']))
            y_predict = []
            for i, sample in enumerate(x_predict):
                sample = torch.from_numpy(np.array([sample]).astype(np.float32))
                sample = Variable(sample)
                y = trained_cnn_net(sample)
                _, y = (torch.max(y, 1))
                y = y.data.numpy()[0][0]
                y_predict.append(y)
            return y_predict
