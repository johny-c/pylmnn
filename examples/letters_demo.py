import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import get_data_home
from pylmnn.lmnn import LMNN
from pylmnn.bayesopt import findLMNNparams
from pylmnn.helpers import test_knn, plot_ba


def fetch_load_data(data_dir=None):
    import csv, os
    path = os.path.join(get_data_home(data_dir), 'letter-recognition.data')

    if not os.path.exists(path):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        print('Downloading letter-recognition dataset from {}...'.format(url))
        request.urlretrieve(url=url, filename=path)
    else:
        print('Found letter-recognition dataset!')

    X, y = [], []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            y.append(row[0])
            X.append(row[1:])
    labels, label_idx = np.unique(y, return_inverse=True)
    return np.asarray(X, dtype=float), label_idx


def main(autotune=True, load=0):
    print('Loading dataset...')
    X, y = fetch_load_data()

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y)
    n, d = xtr.shape
    print('{} images in total'.format(len(ytr) + len(yte)))
    print('{} training images of dimension {}'.format(n, d))

    if autotune:
        # Separate in training and validation set
        xtr, xva, ytr, yva = train_test_split(xtr, ytr, train_size=0.2, stratify=ytr)

        # LMNN hyper-parameter tuning
        print('Searching for optimal LMNN params...\n')
        t_lmnnParams = time()
        Klmnn, Knn, outdim, maxiter = findLMNNparams(xtr, ytr, xva, yva)
        t_bo = time() - t_lmnnParams
        print('Found optimal LMNN params for %d points in %s\n' % (len(ytr), t_bo))

        # Reconstruct full training set
        xtr = np.concatenate((xtr, xva))
        ytr = np.concatenate((ytr, yva))
    else:
        Klmnn, Knn, outdim, maxiter = 1, 1, 16, 82
        # logging levels: DEBUG=10, INFO=20
    lmnn = LMNN(verbose=True, k=Klmnn, max_iter=maxiter, outdim=outdim, save=None, loglevel=10)
    if load == 0:
        # Train full model
        print('Training final model...\n')
        t1 = time()
        lmnn, loss, det = lmnn.fit(xtr, ytr)
        print('LMNN trained in {:.8f}s'.format(time()-t1))
    else:
        lmnn.load_stored(load)

    test_knn(xtr, ytr, xte, yte, k=min(Knn, lmnn.params['k']))
    test_knn(xtr, ytr, xte, yte, k=Knn, L=lmnn.L)
    plot_ba(lmnn.L, xte, yte)


if __name__ == '__main__':
    main(False)
