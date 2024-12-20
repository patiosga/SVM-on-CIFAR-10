import numpy as np
from pickle import load


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = load(fo, encoding='bytes')
    return dict


def read_data(file='data/cifar-10-batches-py/', mask=True):
    batch_data = unpickle(file + 'data_batch_1')  # λεξικό με κλειδιά τα παρακάτω κλειδιά (μαζί με το b'' !!!)
    data = batch_data[b'data']
    labels = batch_data[b'labels']
    for i in range(2, 6):
        file = 'data/cifar-10-batches-py/' + f'data_batch_{str(i)}'
        batch_data = unpickle(file)  # λεξικό με κλειδιά τα παρακάτω κλειδιά (μαζί με το b'' !!!)
        # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        data = np.concatenate((data, batch_data[b'data']), axis=0)
        labels = np.concatenate((labels, batch_data[b'labels']), axis=0)
        # προκύπτουν numpy arrays με shape (50000, 3072) και (50000, 1) αντίστοιχα με το concatenate
    

    # FILTER FOR CLASSES 5, 9 --> fog, truck
    if mask:
        mask = np.isin(labels, [5, 9])
        data = data[mask]
        labels = labels[mask]

    return data, labels


def read_test_data(file='data/cifar-10-batches-py/test_batch', mask=True):
    test_data = unpickle(file)
    # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    X_test, y_test = test_data[b'data'], np.array(test_data[b'labels'])  # data: (10000, 3072), labels: (10000,) numpy arrays

    # FILTER FOR CLASSES 5, 9
    if mask:
        mask = np.isin(y_test, [5, 9])
        X_test = X_test[mask]
        y_test = y_test[mask]
    
    return X_test, y_test


def read_meta_data(file='data/cifar-10-batches-py/batches.meta'):
    meta_data = unpickle(file)
    # {b'num_cases_per_batch': 10000, b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck'], b'num_vis': 3072}
    return meta_data


def main():
    data, labels = read_data()
    print(data.shape)  # (50000, 3072) το value του κλειδιού b'data' είναι numpy array για καθένα από τα 5 batches (το καθένα είναι λεξικό)
    print(labels.shape)  # (50000, 1) το value του κλειδιού b'labels' είναι numpy array για καθένα από τα 5 batches (το καθένα είναι λεξικό)
    # b'...' is the byte encoding of the string και αυτό χρησιμοποιείται ως κλειδί στο dictionary
    print(labels[:10])
    
    test_data, test_labels = read_test_data()

    print(test_data.shape)  # (10000, 3072) το value του κλειδιού b'data' είναι numpy array για το test batch
    print(test_labels.shape)  # (10000,) το value του κλειδιού b'labels' είναι numpy array για το test batch
    print(test_labels[:10])



if __name__ == '__main__':
    main()