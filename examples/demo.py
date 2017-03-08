import numpy as np
from time import time
import logging, sys
from sklearn.model_selection import train_test_split
from configparser import ConfigParser

from pylmnn.bayesopt import find_hyperparams
from pylmnn.lmnn import LargeMarginNearestNeighbor
from pylmnn.helpers import test_knn, plot_ba, pca_transform

from data_fetch import fetch_from_config


def main(demo='shrec14'):

    if demo not in ['shrec14', 'mnist', 'letters', 'usps', 'isolet', 'faces']:
        raise FileNotFoundError('Dataset {} not found in examples directory! Exiting.'.format(demo))

    cfg = ConfigParser()
    cfg.read(demo + '.cfg')
    data_set_name = cfg['fetch']['name']
    print('Data set name: {}'.format(data_set_name))

    x_tr, x_te, y_tr, y_te = fetch_from_config(cfg)

    print('{} features of dimension {}'.format(len(y_tr) + len(y_te), x_tr.shape[1]))

    if cfg['pre_process'].getboolean('pca'):
        print('Cleaning data set...')
        X = pca_transform(np.concatenate((x_tr, x_te)), var_ratio=0.95)
        x_tr, x_te = X[:x_tr.shape[0]], X[x_tr.shape[0]:]

    bo = cfg['bayes_opt']
    if bo.getboolean('perform'):
        # Separate training and validation set
        x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=bo.getfloat('test_size'), stratify=y_tr)

        # Hyper-parameter tuning
        print('Searching for optimal LMNN hyper parameters...\n')
        t_bo = time()
        params = {'log_level': logging.INFO}
        max_trials = bo.getint('max_trials', fallback=12)
        k_tr, k_te, dim_out, max_iter = find_hyperparams(x_tr, y_tr, x_va, y_va, params, max_trials=max_trials)
        print('Found optimal LMNN hyper parameters for %d points in %s\n' % (len(y_tr), time() - t_bo))

        # Reconstruct full training set
        x_tr = np.concatenate((x_tr, x_va))
        y_tr = np.concatenate((y_tr, y_va))
    else:
        hyper_params = cfg['hyper_params']
        k_tr = hyper_params.getint('k_tr')
        k_te = hyper_params.getint('k_te')
        dim_out = hyper_params.getint('dim_out')
        max_iter = hyper_params.getint('max_iter')

    clf = LargeMarginNearestNeighbor(k=k_tr, max_iter=max_iter, dim_out=dim_out, log_level=logging.INFO)

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

    test_knn(x_tr, y_tr, x_te, y_te, k=min(k_te, clf.k))
    test_knn(x_tr, y_tr, x_te, y_te, k=k_te, L=clf.L)
    plot_ba(clf.L, x_te, y_te)


if __name__ == '__main__':
    main(str(sys.argv[1])) if len(sys.argv) > 1 else main()

