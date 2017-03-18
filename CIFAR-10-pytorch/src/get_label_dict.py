import pickle

def unpickle(file):
    fo = open(file, 'rb')
    data_dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return data_dict

if __name__ == '__main__':
    file = '../database/batches.meta'
    data = unpickle(file)
    print(data)
    label_names = data['label_names']
    with open('../database/CIFAR_10-Label-Names.pkl', 'wb') as file:
        pickle.dump(label_names, file)
