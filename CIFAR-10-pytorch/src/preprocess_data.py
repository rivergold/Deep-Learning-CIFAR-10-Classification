'''
This script convert orginal CIFAR-10 dataset into .pkl
    - train.pkl
    - test.pkl
when you run this script, you need to input argument as [-m] [--mode] train-set | test-set
'''
import pickle
import numpy as np
import cv2
import argparse


def unpickle(file):
    fo = open(file, 'rb')
    data_dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return data_dict

if __name__ == '__main__':
    # Parse input arguments
    arg_parser = argparse.ArgumentParser(
        description='Preprocess train or test set')
    arg_parser.add_argument(
        '-m', '--mode', choices=['train-set', 'test-set'], help='input which set you want to preprocess')
    args = arg_parser.parse_args()
    #
    if args.mode == 'train-set':
        save_data = {'data': [], 'label': []}
        for file_id in range(1, 6):
            file = 'F:/Windfall/Common Dataset/cifar-10-batches-py/data_batch_{}'.format(
                file_id)
            data = unpickle(file)
            for i in range(len(data['labels'])):
                image_data = np.array(data['data'][i])
                r = image_data[0:1024].reshape(32, 32)
                g = image_data[1024:2048].reshape(32, 32)
                b = image_data[2048:].reshape(32, 32)
                #
                sample_data = np.array([r, g, b])
                label = data['labels'][i]
                #
                save_data['data'].append(sample_data)
                save_data['label'].append(label)
                # =================================
                # Show image.
                # image_data = cv2.merge([b, g, r])
                # input(image_data.shape)
                # image_data = cv2.resize(image_data, (128, 128))
                # cv2.imshow('image', image_data)
                # cv2.waitKey(0)
                # input('lable is {}'.format(data['labels'][i]))
                # =================================
        # Save.
        with open('./CIFAR-10-Train.pkl', 'wb') as pickle_file:
            pickle.dump(save_data, pickle_file)
    #
    elif args.mode == 'test-set':
        save_data = {'data': [], 'label': []}
        file = 'F:/Windfall/Common Dataset/cifar-10-batches-py/test_batch'
        data = unpickle(file)
        for i in range(len(data['labels'])):
            image_data = np.array(data['data'][i])
            r = image_data[0:1024].reshape(32, 32)
            g = image_data[1024:2048].reshape(32, 32)
            b = image_data[2048:].reshape(32, 32)
            #
            sample_data = np.array([r, g, b])
            label = data['labels'][i]
            #
            save_data['data'].append(sample_data)
            save_data['label'].append(label)
            # ===========================================
            # Show image.
            # image_data = cv2.merge([b, g, r])
            # input(image_data.shape)
            # image_data = cv2.resize(image_data, (128, 128))
            # cv2.imshow('image', image_data)
            # cv2.waitKey(24)
            # input('lable is {}'.format(data['labels'][i]))
            # ============================================
        # Save.
        with open('./CIFAR-10-Test.pkl', 'wb') as pickle_file:
            pickle.dump(save_data, pickle_file)
