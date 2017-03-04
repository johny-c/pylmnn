import numpy as np
from time import time
import logging
from sklearn.model_selection import train_test_split
from configparser import ConfigParser

from pylmnn.bayesopt import find_hyper_params
from pylmnn.lmnn import LargeMarginNearestNeighbor
from pylmnn.helpers import test_knn, plot_ba, pca_transform

from data_fetch import fetch_from_config


def main(demo='mnist'):

    cfg = ConfigParser()
    cfg.read(demo + '.cfg')

    x_tr, x_te, y_tr, y_te = fetch_from_config(cfg)

    print('{} features of dimension {}'.format(len(y_tr) + len(y_te), x_tr.shape[1]))

    if cfg['pre_process'].getboolean('pca'):
        print('Cleaning data set...')
        X = pca_transform(np.concatenate((x_tr, x_te)), var_ratio=0.95)
        x_tr, x_te = X[:x_tr.shape[0]], X[x_tr.shape[0]:]

    bo = cfg['bayes_opt']
    if bo.getboolean('perform'):
        # Separate training and validation set
        test_size = bo.getfloat('test_size')
        x_tr, xva, y_tr, yva = train_test_split(x_tr, y_tr, test_size=test_size, stratify=y_tr)

        # Hyper-parameter tuning
        print('Searching for optimal LMNN hyper parameters...\n')
        t_bo = time()
        params = {'log_level': logging.DEBUG}
        k_tr, k_te, dim_out, max_iter = find_hyper_params(x_tr, y_tr, xva, yva,
                                                          params, bo['max_trials'])
        print('Found optimal LMNN hyper parameters for %d points in %s\n' % (len(y_tr), time() - t_bo))

        # Reconstruct full training set
        x_tr = np.concatenate((x_tr, xva))
        y_tr = np.concatenate((y_tr, yva))
    else:
        hyper_params = cfg['hyper_params']
        k_tr = hyper_params.getint('k_tr')
        k_te = hyper_params.getint('k_te')
        dim_out = hyper_params.getint('dim_out')
        max_iter = hyper_params.getint('max_iter')

    clf = LargeMarginNearestNeighbor(k=k_tr, max_iter=max_iter, dim_out=dim_out)

    # Train full model
    t_train = time()
    print('Training final model...\n')
    clf = clf.fit(x_tr, y_tr)

    t_train = time() - t_train
    print('\nStatistics:\n{}\nLMNN trained in: {:.4f} s'.format('-'*50, t_train))
    print('Number of iterations: {}'.format(clf.details['nit']))
    print('Number of function calls: {}'.format(clf.details['funcalls']))
    print('Average time / function call: {:.4f} s'.format(t_train / clf.details['funcalls']))
    print('Training loss: {}'.format(clf.details['loss']))

    test_knn(x_tr, y_tr, x_te, y_te, k=min(k_te, clf.params['k']))
    test_knn(x_tr, y_tr, x_te, y_te, k=k_te, L=clf.L)
    plot_ba(clf.L, x_te, y_te)


if __name__ == '__main__':
    main()
