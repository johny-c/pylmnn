from time import time
import numpy as np

from pylmnn.bayesopt import find_hyper_params
from pylmnn.lmnn import LargeMarginNearestNeighbor
from pylmnn.helpers import test_knn, plot_ba

import sklearn.datasets as skd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse

# Out of memory
def main(autotune=True, load=0):
    print('Loading dataset...')
    news_train = skd.fetch_20newsgroups_vectorized(subset='train', remove=('headers', 'footers', 'quotes'))
    news_test  = skd.fetch_20newsgroups_vectorized(subset='test', remove=('headers', 'footers', 'quotes'))

    X = sparse.vstack((news_train.data, news_test.data))
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(X)
    ytr, yte = news_train.target, news_test.target

    print('Cleaning dataset...')
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    X = svd.fit_transform(X)  # X is now a dense numpy array

    print('Explained variance ratio: {}'.format(sum(svd.explained_variance_ratio_)))
    xtr, xte = X[:len(ytr)], X[len(ytr):]

    # xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y)
    n, d = xtr.shape
    print('{} training samples of dimension {}'.format(n, d))

    if autotune:
        # Separate in training and validation set
        xtr, xva, ytr, yva = train_test_split(xtr, ytr, train_size=0.1, stratify=ytr)

        # LMNN hyper-parameter tuning
        print('Searching for optimal LMNN params...\n')
        t_lmnnParams = time()
        Klmnn, Knn, outdim, maxiter = find_hyper_params(xtr, ytr, xva, yva)
        t_bo = time() - t_lmnnParams
        print('Found optimal LMNN params for %d points in %s\n' % (len(ytr), t_bo))

        # Reconstruct full training set
        xtr = np.concatenate((xtr, xva))
        ytr = np.concatenate((ytr, yva))
    else:
        Klmnn, Knn, outdim, maxiter = 15, 3, 28, 25

    lmnn = LargeMarginNearestNeighbor(verbose=True, k=Klmnn, max_iter=maxiter, dim_out=outdim, save=None, log_level=10)
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
