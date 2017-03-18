import pickle

def unpickle(file):
    fo = open(file, 'rb')
    data_dict = pickle.load(fo)
    fo.close()
    return data_dict

if __name__ == '__main__':
    file = './CIFAR-10-Train.pkl'
    data = unpickle(file)
    input(len(data['data']))
