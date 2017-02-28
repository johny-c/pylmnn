import numpy as np
from time import time
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split
from pylmnn.lmnn import LMNN
from pylmnn.bayesopt import findLMNNparams
from pylmnn.helpers import test_knn, plot_ba, clean_data


def main(autotune=True, load=0):
    print('Loading dataset...')
    dataset = skd.fetch_mldata('USPS')
    X, y = dataset.data, dataset.target

    print('Cleaning dataset...')
    X = clean_data(X, var_ratio=0.95)

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y)
    n, d = xtr.shape
    print('{} images in total'.format(len(ytr) + len(yte)))
    print('{} training images of dimension {}'.format(n, d))

    if autotune:
        # Separate in training and validation set
        xtr, xva, ytr, yva = train_test_split(xtr, ytr, train_size=0.25, stratify=ytr)

        # LMNN hyper-parameter tuning
        print('Searching for optimal LMNN params...\n')
        t_lmnnParams = time()
        Klmnn, Knn, outdim, maxiter = findLMNNparams(xtr, ytr, xva, yva, max_trials=20)
        t_bo = time() - t_lmnnParams
        print('Found optimal LMNN params for %d points in %s\n' % (len(ytr), t_bo))

        # Reconstruct full training set
        xtr = np.concatenate((xtr, xva))
        ytr = np.concatenate((ytr, yva))
    else:
        Klmnn, Knn, outdim, maxiter = 14, 4, 20, 187

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
    main(True)
