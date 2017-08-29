import numpy as np
from sklearn.datasets import get_data_home
from scipy.io import loadmat, savemat
import os
from time import time
from sklearn.neighbors import KNeighborsClassifier, LargeMarginNearestNeighbor
from sklearn.decomposition import PCA
import deskewing


MNIST_ORIGINAL_PATH = os.path.join(get_data_home(), 'mldata', 'mnist-original.mat')
MNIST_DESKEWED_URL = 'https://www.dropbox.com/s/mhsnormwt5i2ba6/mnist-deskewed-pca164.mat?dl=1'
MNIST_DESKEWED_PATH = os.path.join(get_data_home(), 'mnist-deskewed-pca164.mat')
MNIST_TARGETS_PATH = os.path.join(get_data_home(), 'mnist-targets.mat')

CWD = os.path.split(__file__)[0]
TRANSFORMATIONS_DIR = os.path.join(CWD, 'MNIST_ORIG_DESKEWED_TRANSFORMATIONS')

LMNN_PARAMS = {'n_neighbors': 3,
               'max_iter': 120,
               'n_features_out': 164,
               'verbose': 1,
               'random_state': 42,
               'n_jobs': -1}


def deskew_mnist(path=MNIST_ORIGINAL_PATH):
    """

    Parameters
    ----------
    path : path to the original MNIST data.
        File contains a dictionary with keys-values:
        * data: array, shape (784, 70000), dtype uint8
        * label: array, shape (1, 70000), dtype <f8

    Returns
    -------
    X: array, shape (70000, 784)
        Deskewed images.
    y: array, shape (70000,)
        The labels.
    """

    mnist_mat = loadmat(path)
    X = mnist_mat['data']
    X = np.asarray(X, dtype=np.float64).T / 255.

    y = mnist_mat['label']
    y = np.asarray(y, dtype=np.int).ravel()

    print('Deskewing dataset... ', end='', flush=True)
    t = time()
    for i in range(len(X)):
        X[i] = deskewing.deskew(X[i].reshape(28, 28)).ravel()

    print('done in {:8.2f}s'.format(time() - t))

    return X, y


def save_deskewed_mnist_pca(path=MNIST_ORIGINAL_PATH, n_components=164,
                            random_state=42):

    X, y = deskew_mnist(path)

    print('Performing PCA... ', end='', flush=True)
    t = time()
    pca = PCA(n_components=n_components, random_state=random_state)
    X = pca.fit_transform(X)
    print('done in {:8.2f}s'.format(time() - t))

    X_train = X[:60000]
    X_test = X[60000:]
    y_train = y[:60000]
    y_test = y[60000:]

    new_mat = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test,
               'y_test': y_test}

    name = 'mnist-deskewed-pca' + str(n_components) + '.mat'
    new_path = os.path.join(get_data_home(), name)
    savemat(new_path, new_mat, oned_as='column')


def fetch_deskewed_mnist(url=MNIST_DESKEWED_URL):

    if not os.path.exists(MNIST_DESKEWED_PATH):
        from urllib import request
        request.urlretrieve(url, MNIST_DESKEWED_PATH)

    mnist_mat = loadmat(MNIST_DESKEWED_PATH)

    X_train = np.asarray(mnist_mat['X_train'], dtype=np.float64).T
    X_test = np.asarray(mnist_mat['X_test'], dtype=np.float64).T
    y_train = np.asarray(mnist_mat['y_train'], dtype=np.int).ravel()
    y_test = np.asarray(mnist_mat['y_test'], dtype=np.int).ravel()

    return X_train, y_train, X_test, y_test


def save_cb(transformation, iteration):
    filename = 't_{:03d}.npy'.format(iteration)
    filepath = os.path.join(TRANSFORMATIONS_DIR, filename)
    np.save(filepath, transformation)


def train():

    X_train, y_train, X_test, y_test = fetch_deskewed_mnist()

    if not os.path.exists(TRANSFORMATIONS_DIR):
        os.makedirs(TRANSFORMATIONS_DIR)

    if os.path.exists(MNIST_TARGETS_PATH):
        mnist_targets_mat = loadmat(MNIST_TARGETS_PATH)
        LMNN_PARAMS['targets'] = mnist_targets_mat['idx']

    LMNN_PARAMS['callback'] = save_cb

    lmnn = LargeMarginNearestNeighbor(**LMNN_PARAMS)

    t_train = time()
    lmnn.fit(X_train, y_train)
    t_train = time() - t_train
    print('Finished training in {:8.2f}s'.format(t_train))


def test(transformation_file=None):

    if transformation_file is None:
        # Load the last transformation written on disk
        tr_files = os.listdir(TRANSFORMATIONS_DIR)
        tr_files = [os.path.join(TRANSFORMATIONS_DIR, f) for f in tr_files]
        transformation_file = max(tr_files, key=os.path.getctime)

    if not os.path.exists(transformation_file):
        raise FileNotFoundError('{} not found!'.format(transformation_file))

    transformation = np.load(transformation_file)
    X_train, y_train, X_test, y_test = fetch_deskewed_mnist()

    dim_in = X_train.shape[1]
    dim_out = transformation.size // dim_in
    transformation = transformation.reshape(dim_out, dim_in)
    print('Transformation shape: {}'.format(transformation.shape))
    dim_tran = transformation.shape[1]
    if dim_in != dim_tran:
        raise ValueError('Transformation - data dimensionality mismatch: {} '
                         '- {}'.format(dim_tran, dim_in))

    print('\nNow testing . . .')
    knn = KNeighborsClassifier(n_neighbors=LMNN_PARAMS['n_neighbors'],
                               n_jobs=-1)
    t_train = time()
    print('Fitting nearest neighbor with learned transformation...')
    knn.fit(X_train.dot(transformation.T), y_train)
    t_train = time() - t_train
    print('Done in {:8.2f}s, now testing...'.format(t_train))

    t_test = time()
    test_acc = knn.score(X_test.dot(transformation.T), y_test)
    t_test = time() - t_test
    print('Finished testing in {:8.2f}s'.format(t_test))
    print('LMNN accuracy of transformation {}: {:5.2f}%'.format(
        transformation_file, test_acc*100))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LMNN demo on deskewed MNIST')

    parser.add_argument('-tr',
        '--train', type=int, default=1,
        help='Training mode (default: %(default)s)')
    parser.add_argument('-te',
        '--test', type=int, default=0,
        help='Testing mode (default: %(default)s)')
    parser.add_argument('-tf',
        '--tfile', type=str, default=None,
        help='Transformation file to test with (default: %(default)s)')

    args = parser.parse_args()
    if args.train:
        train()

    if args.test:
        test(args.test)

