import os
import sys
import csv
import yaml
import numpy as np
from sklearn.datasets import get_data_home, fetch_olivetti_faces, \
    fetch_mldata, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors.lmnn import LargeMarginNearestNeighbor as LMNN


def decompress_z(fname_in, fname_out=None):
    from unlzw import unlzw
    fname_out = fname_in[:-2] if fname_out is None else fname_out
    print('Extracting {} to {}...'.format(fname_in, fname_out))
    with open(fname_in, 'rb') as fin, open(fname_out, 'wb') as fout:
        compressed_data = fin.read()
        uncompressed_data = unlzw(compressed_data)
        fout.write(uncompressed_data)


def fetch_isolet(data_dir=None):
    train = 'isolet1+2+3+4.data.Z'
    test = 'isolet5.data.Z'
    path_train = os.path.join(get_data_home(data_dir), train)
    path_test = os.path.join(get_data_home(data_dir), test)

    if not os.path.exists(path_train[:-2]) or not os.path.exists(
            path_test[:-2]):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/'
        if not os.path.exists(path_train[:-2]):
            if not os.path.exists(path_train):
                print(
                    'Downloading Isolated Letter Speech Recognition data set from {}...'.format(
                        url))
                request.urlretrieve(url=url + train, filename=path_train)
            # os.system('gzip -d ' + path_train)
            decompress_z(path_train)
        if not os.path.exists(path_test[:-2]):
            if not os.path.exists(path_test):
                print(
                    'Downloading Isolated Letter Speech Recognition data set from {}...'.format(
                        url))
                request.urlretrieve(url=url + test, filename=path_test)
            # os.system('gzip -d ' + path_test)
            decompress_z(path_test)
    else:
        print('Found Isolated Letter Speech Recognition data set!')

    X_train, y_train = [], []
    with open(path_train[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            X_train.append(row[:-1])
            y_train.append(int(float(row[-1])))

    labels, y_train = np.unique(y_train, return_inverse=True)

    X_test, y_test = [], []
    with open(path_test[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            X_test.append(row[:-1])
            y_test.append(int(float(row[-1])))

    labels, y_test = np.unique(y_test, return_inverse=True)

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    return X_train, y_train, X_test, y_test


def fetch_usps(save_dir=None):

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'
    train_file = 'zip.train.gz'
    test_file = 'zip.test.gz'
    save_dir = get_data_home() if save_dir is None else save_dir

    if not os.path.isdir(save_dir):
        raise NotADirectoryError('{} is not a directory.'.format(save_dir))

    def download_file(source, destination):

        if not os.path.exists(destination):
            from urllib import request
            print('Downloading dataset from {}...'.format(source))
            f, msg = request.urlretrieve(url=source, filename=destination)
            print('HTTP response: {}'.format(msg))
        else:
            print('Found dataset in {}!'.format(destination))

    train_source = os.path.join(base_url, train_file)
    test_source = os.path.join(base_url, test_file)

    train_dest = os.path.join(save_dir, train_file)
    test_dest = os.path.join(save_dir, test_file)

    download_file(train_source, train_dest)
    download_file(test_source, test_dest)

    X_train = np.loadtxt(train_dest)
    y_train, X_train = X_train[:, 0].astype(np.int32), X_train[:, 1:]

    X_test = np.loadtxt(test_dest)
    y_test, X_test = X_test[:, 0].astype(np.int32), X_test[:, 1:]

    return X_train, y_train, X_test, y_test


def fetch_data(dataset, split=True):

    if dataset == 'isolet':
        X_train, y_train, X_test, y_test = fetch_isolet()
        if split:
            return X_train, y_train, X_test, y_test
        else:
            return np.concatenate((X_train, X_test)), \
                   np.concatenate((y_train, y_test))
    elif dataset == 'usps':
        X_train, y_train, X_test, y_test = fetch_usps()
        if split:
            return X_train, y_train, X_test, y_test
        else:
            return np.concatenate((X_train, X_test)), \
                   np.concatenate((y_train, y_test))
    elif dataset == 'olivetti_faces':
        data = fetch_olivetti_faces(shuffle=True)
        if split:
            X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                         data.target,
                                                      stratify=data.target,
                                                      test_size=0.3)
            return X_train, y_train, X_test, y_test
        else:
            return data.data, data.target
    elif dataset == 'iris':
        data = load_iris(True)
        X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                            data.target,
                                                      stratify=data.target,
                                                      test_size=0.3)
        if split:
            return X_train, y_train, X_test, y_test
        else:
            return data.data, data.target
    elif dataset == 'mnist':
        data = fetch_mldata('MNIST original')
        X = data.data
        # normalize to [0, 1]
        X = X / 255.
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = data.target[:60000], data.target[60000:]
        return X_train, y_train, X_test, y_test
    else:
        raise NotImplementedError('Unknown dataset {}!'.format(dataset))


def benchmark(dataset):

    with open('lmnn_params.yml', 'r') as config_file:
        params_dict = yaml.load(config_file)

    datasets = params_dict.keys()

    if dataset not in datasets:
        raise NotImplementedError('Currently supported:\n{}'.format(datasets))

    params = params_dict[dataset]
    n_splits = params.get('n_splits', 1)
    lmnn_params = params['lmnn_params']

    if n_splits == 1:
        X_train, y_train, X_test, y_test = fetch_data(dataset)
    else:
        X, y = fetch_data(dataset, split=False)

    for i in range(n_splits):



def main(argv):
    # usage python3 lmnn_demo_usps.py isolet

    dataset = argv[1] if len(argv) > 1 else 'isolet'
    X_train, y_train, X_test, y_test = fetch_data(dataset)

    knn_clf = KNN(n_neighbors=3)
    knn_clf.fit(X_train, y_train)
    knn_err = 1. - knn_clf .score(X_test, y_test)
    print('KNN Test error on {}: {:5.2f}%'.format(dataset, knn_err*100))

    with open('lmnn_params.yml', 'r') as config_file:
        lmnn_params = yaml.load(config_file)
    clf = LMNN(verbose=1, **lmnn_params[dataset])
    clf.fit(X_train, y_train)
    lmnn_err = 1. - clf.score(X_test, y_test)
    print('LMNN Test error on {}: {:5.2f}%'.format(dataset, lmnn_err*100))
    print('KNN  Test error on {}: {:5.2f}%'.format(dataset, knn_err * 100))


if __name__ == '__main__':
    main(sys.argv)