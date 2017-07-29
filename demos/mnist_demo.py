import numpy as np
from sklearn.datasets import get_data_home
from scipy.io import loadmat
import os
from time import time
from sklearn.neighbors import KNeighborsClassifier, LargeMarginNearestNeighbor
from sklearn.decomposition import PCA
from deskewing import deskew


MNIST_PATH = os.path.join(get_data_home(), 'mldata', 'mnist-original.mat')
CWD = os.path.split(__file__)[0]
TRANSFORMATIONS_DIR = os.path.join(CWD, 'MNIST_ORIG_TRANSFORMATIONS')


def fetch_mnist(path=MNIST_PATH, deskewed=True):
    mnist_mat = loadmat(path)

    X = mnist_mat['data']
    X = np.asarray(X, dtype=np.float64).T / 255.

    y = mnist_mat['label']
    y = np.asarray(y, dtype=np.int).ravel()

    if deskewed:
        print('Deskewing dataset... ', end='', flush=True)
        t = time()
        for i in range(len(X)):
            X[i] = deskew(X[i].reshape(28, 28)).ravel()

        print('done in {:8.2f}s'.format(time()-t))

    print('Performing PCA... ', end='', flush=True)
    t = time()
    pca = PCA(n_components=LMNN_PARAMS['n_features_out'],
              random_state=LMNN_PARAMS['random_state'])
    X = pca.fit_transform(X)
    print('done in {:8.2f}s'.format(time()-t))

    X_train = X[:60000]
    X_test = X[60000:]
    y_train = y[:60000]
    y_test = y[60000:]

    return X_train, y_train, X_test, y_test


def save_cb(transformation, iteration):
    filename = 't_{:03d}.npy'.format(iteration)
    filepath = os.path.join(TRANSFORMATIONS_DIR, filename)
    np.save(filepath, transformation)


LMNN_PARAMS = {'n_neighbors': 3, 'max_iter': 120, 'n_features_out': 164,
               'verbose': 1, 'random_state': 42, 'callback': save_cb,
               'n_jobs':-1}


def train(testing=True):

    X_train, y_train, X_test, y_test = fetch_mnist()

    if not os.path.exists(TRANSFORMATIONS_DIR):
        os.makedirs(TRANSFORMATIONS_DIR)

    lmnn = LargeMarginNearestNeighbor(**LMNN_PARAMS)

    t_train = time()
    lmnn.fit(X_train, y_train)
    t_train = time() - t_train
    print('Finished training in {:8.2f}s'.format(t_train))

    if testing:
        test()


def test(iteration=None):

    print('\nNow testing . . .')
    X_train, y_train, X_test, y_test = fetch_mnist()

    knn = KNeighborsClassifier(n_neighbors=LMNN_PARAMS['n_neighbors'],
                               n_jobs=-1)

    if iteration is None:
        tr_files = os.listdir(TRANSFORMATIONS_DIR)
        tr_files = [os.path.join(TRANSFORMATIONS_DIR, f) for f in tr_files]
        transformation_file = max(tr_files, key=os.path.getctime)
    else:
        filename = 't_{:03d}.npy'.format(iteration)
        transformation_file = os.path.join(TRANSFORMATIONS_DIR, filename)

    transformation = np.load(transformation_file)
    dim_out = transformation.size // X_train.shape[1]
    transformation = transformation.reshape(dim_out, X_train.shape[1])
    print('Transformation shape: {}'.format(transformation.shape))
    dim_data = X_train.shape[1]
    dim_tran = transformation.shape[1]
    if dim_data != dim_tran:
        raise ValueError('Transformation - data dimensionality mismatch: {} '
                         '- {}'.format(dim_tran, dim_data))

    t_train = time()
    print('Fitting nearest neighbor with learned transformation...')
    knn.fit(X_train.dot(transformation.T), y_train)
    t_train = time() - t_train
    print('Finished training in {:8.2f}s, now testing...'.format(t_train))

    t_test = time()
    test_acc = knn.score(X_test.dot(transformation.T), y_test)
    t_test = time() - t_test
    print('Finished testing in {:8.2f}s'.format(t_test))
    print('LMNN accuracy of transformation {}: {:5.2f}%'.format(
        transformation_file, test_acc*100))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LMNN demo on MNIST')

    parser.add_argument('-tr',
        '--train', type=int, default=1,
        help='Training mode (default: %(default)s)')
    parser.add_argument('-it',
        '--test', type=int, default=0,
        help='Transformation iteration to test with (default: %(default)s)')

    args = parser.parse_args()
    if args.test != 0:
        test(args.test)
    else:
        train()
