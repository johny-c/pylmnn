import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import get_data_home
from pylmnn.lmnn import LMNN
from pylmnn.bayesopt import findLMNNparams
from pylmnn.helpers import test_knn, plot_ba, clean_data


def fetch_load_data(data_dir=None):
    import csv, os
    train = 'isolet1+2+3+4.data.Z'
    test = 'isolet5.data.Z'
    path_train = os.path.join(get_data_home(data_dir), train)
    path_test  = os.path.join(get_data_home(data_dir), test)

    if not os.path.exists(path_train) or not os.path.exists(path_test):
        from urllib import request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/'
        print('Downloading Isolated Letter Speech Recognition dataset from {}...'.format(url))
        if not os.path.exists(path_train):
            request.urlretrieve(url=url+train, filename=path_train)
            os.system('gzip -d ' + path_train)
        if not os.path.exists(path_test):
            request.urlretrieve(url=url+test, filename=path_test)
            os.system('gzip -d ' + path_test)
    else:
        print('Found Isolated Letter Speech Recognition dataset!')

    xtr, ytr = [], []
    with open(path_train[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            xtr.append(row[:-1])
            ytr.append(int(float(row[-1])))
    labels, ytr = np.unique(ytr, return_inverse=True)
    xte, yte = [], []
    with open(path_test[:-2]) as f:
        reader = csv.reader(f)
        for row in reader:
            xte.append(row[:-1])
            yte.append(int(float(row[-1])))
    labels, yte = np.unique(yte, return_inverse=True)

    return np.asarray(xtr, dtype=float), np.asarray(xte, dtype=float), ytr, yte


def main(autotune=True, load=0):
    print('Loading dataset...')
    xtr, xte, ytr, yte = fetch_load_data()
    n, d = xtr.shape
    print('{} images in total'.format(len(ytr) + len(yte)))
    print('{} training images of dimension {}'.format(n, d))

    print('Cleaning dataset...')
    X = clean_data(np.concatenate((xtr, xte)), var_ratio=0.95)
    xtr, xte = X[:xtr.shape[0]], X[xtr.shape[0]:]

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
        Klmnn, Knn, outdim, maxiter = 15, 7, 46, 180

    lmnn = LMNN(verbose=True, k=Klmnn, max_iter=maxiter, outdim=outdim, save=None)
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
