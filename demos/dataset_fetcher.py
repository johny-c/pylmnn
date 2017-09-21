import os
import csv
import numpy as np
from scipy.io import loadmat
from sklearn.datasets import get_data_home, fetch_olivetti_faces, \
    fetch_mldata, load_iris
from sklearn.model_selection import train_test_split


def fetch_letters(data_dir=None):
    path = os.path.join(get_data_home(data_dir), 'letter-recognition.data')

    if not os.path.exists(path):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        print('Downloading letter-recognition dataset from {}...'.format(url))
        request.urlretrieve(url=url, filename=path)
    else:
        print('Found letter-recognition in {}!'.format(path))

    y = np.loadtxt(path, dtype=str, usecols=(0), delimiter=',')
    X = np.loadtxt(path, usecols=range(1, 17), delimiter=',')

    return X, y, None, None


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


USPS_DIR = os.path.join(get_data_home(), 'usps')


def fetch_usps(save_dir=USPS_DIR):

    # base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'
    base_url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/'

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


def fetch_mnistPCA(data_dir=None):

    path = os.path.join(get_data_home(data_dir), 'mnistPCA.mat')
    if not os.path.exists(path):
        from urllib import request
        url = 'https://dl.dropboxusercontent.com/u/4284723/DATA/mnistPCA.mat'
        print('Downloading mnistPCA dataset from {}...'.format(url))
        request.urlretrieve(url=url, filename=path)
    else:
        print('Found mnistPCA.mat in {}!'.format(path))

    mnist_mat = loadmat(path)

    X_train = np.asarray(mnist_mat['xTr'], dtype=np.float64)
    X_test = np.asarray(mnist_mat['xTe'], dtype=np.float64)
    y_train = np.asarray(mnist_mat['yTr'], dtype=np.int).ravel()
    y_test = np.asarray(mnist_mat['yTe'], dtype=np.int).ravel()

    return X_train, y_train, X_test, y_test


def fetch_mnist_deskewed(data_dir=None):

    MNIST_DESKEWED_URL = 'https://www.dropbox.com/s/mhsnormwt5i2ba6/mnist-deskewed-pca164.mat?dl=1'
    MNIST_DESKEWED_PATH = os.path.join(get_data_home(),
                                       'mnist-deskewed-pca164.mat')

    if not os.path.exists(MNIST_DESKEWED_PATH):
        from urllib import request
        print('Downloading deskewed MNIST from {} . . .'.format(
            MNIST_DESKEWED_URL), end='')
        request.urlretrieve(MNIST_DESKEWED_URL, MNIST_DESKEWED_PATH)
        print('done.')

    mnist_mat = loadmat(MNIST_DESKEWED_PATH)

    X_train = np.asarray(mnist_mat['X_train'], dtype=np.float64)
    X_test = np.asarray(mnist_mat['X_test'], dtype=np.float64)
    y_train = np.asarray(mnist_mat['y_train'], dtype=np.int).ravel()
    y_test = np.asarray(mnist_mat['y_test'], dtype=np.int).ravel()

    print('Loaded deskewed MNIST from {}.'.format(MNIST_DESKEWED_PATH))

    return X_train, y_train, X_test, y_test


def fetch_dataset(dataset):

    if dataset == 'isolet':
        return fetch_isolet()
    elif dataset == 'usps':
        return fetch_usps()
    elif dataset == 'olivetti_faces':
        faces = fetch_olivetti_faces(shuffle=True)
        return faces.data, faces.target, None, None
    elif dataset == 'letters':
        return fetch_letters()
    elif dataset == 'iris':
        X, y = load_iris(return_X_y=True)
        return X, y, None, None
    elif dataset == 'mnist_deskewed':
        return fetch_mnist_deskewed()
    elif dataset == 'mnistPCA':
        return fetch_mnistPCA()
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
