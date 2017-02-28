import numpy as np
from time import time
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split

from pylmnn.bayesopt import findLMNNparams
from pylmnn.lmnn import LMNN
from pylmnn.helpers import test_knn, plot_ba, clean_data


def main(autotune=True, load=0):
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
        xtr, xva, ytr, yva = train_test_split(xtr, ytr, train_size=0.05, stratify=ytr)

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
        Klmnn, Knn, outdim, maxiter = 3, 3, 300, 120
    # loglevel: DEBUG=10, INFO=20
    lmnn = LMNN(verbose=True, k=Klmnn, max_iter=maxiter, outdim=outdim, save='lin_transf',
                loglevel=10)
    if load == 0:
        # Train full model
        print('Training final model...\n')
        t1 = time()
        lmnn, loss, det = lmnn.fit(xtr, ytr)
        print('LMNN fit in {:.8f}s'.format(time()-t1))
    else:
        lmnn.load_stored(load)

    test_knn(xtr, ytr, xte, yte, k=min(Knn, lmnn.params['k']))
    test_knn(xtr, ytr, xte, yte, k=Knn, L=lmnn.L)
    plot_ba(lmnn.L, xte, yte)


if __name__ == '__main__':
    main(False)


