import numpy as np
from time import time
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split

from pylmnn.lmnn import LargeMarginNearestNeighbor
from pylmnn.bayesopt import find_hyper_params
from pylmnn.helpers import test_knn, plot_ba, clean_data


def main(autotune=True, load=None):
    print('Loading dataset...')
    mnist = skd.fetch_mldata("MNIST original")
    # rescale the data, use the traditional train/test split
    X, y = mnist.data / 255., mnist.target
    xtr, xte = X[:60000], X[60000:]
    ytr, yte = y[:60000], y[60000:]

    print('Cleaning dataset...')
    X = clean_data(np.concatenate((xtr, xte)), var_ratio=0.95)
    xtr, xte = X[:xtr.shape[0]], X[xtr.shape[0]:]

    n, d = xtr.shape
    print('{} images in total'.format(len(ytr) + len(yte)))
    print('{} training images of dimension {}'.format(n, d))

    if autotune:
        # Separate in training and validation set
        xtr, xva, ytr, yva = train_test_split(xtr, ytr, train_size=0.2, stratify=ytr)

        # LMNN hyper-parameter tuning
        print('Searching for optimal LMNN params for {} points...\n'.format(len(xtr)))
        t_lmnnParams = time()
        Klmnn, Knn, outdim, maxiter = find_hyper_params(xtr, ytr, xva, yva, max_trials=50)
        t_bo = time() - t_lmnnParams
        print('Found optimal LMNN params for %d points in %s s\n' % (len(ytr), t_bo))

        # Reconstruct full training set
        xtr = np.concatenate((xtr, xva))
        ytr = np.concatenate((ytr, yva))
    else:
        Klmnn, Knn, outdim, maxiter = 9, 5, 147, 167
        # 3, 3, 300, 120  # (found by K.W.)
        # 9, 5, 147, 67  # (found after 50 runs with 20% training set)
        # 14, 3, 34, 140  # (found after 12 runs with 5% training set)
        # 13, 10, 30, 178  # (found after 12 runs with 5% training set and discrete domain)


    # log_level: DEBUG=10, INFO=20
    lmnn = LargeMarginNearestNeighbor(verbose=True, k=Klmnn, max_iter=maxiter, dim_out=outdim, save='lin_transf', log_level=10, load=load)
    if load is None:
        # Train full model
        print('Training final model...\n')
        t1 = time()
        lmnn, loss, det = lmnn.fit(xtr, ytr)
        print('LMNN fit in {:.8f}s'.format(time()-t1))

    test_knn(xtr, ytr, xte, yte, k=min(Knn, lmnn.params['k']))
    test_knn(xtr, ytr, xte, yte, k=Knn, L=lmnn.L)
    plot_ba(lmnn.L, xte, yte)


if __name__ == '__main__':
    main(True)


