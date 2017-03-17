import sys
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from configparser import ConfigParser

from pylmnn import bayesopt, lmnn, helpers, plots
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
        X = helpers.pca_fit(np.concatenate((x_tr, x_te)), var_ratio=0.95)
        x_tr, x_te = X[:x_tr.shape[0]], X[x_tr.shape[0]:]

    bo = cfg['bayes_opt']
    if bo.getboolean('perform'):
        # Separate training and validation set
        x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=bo.getfloat('test_size'), stratify=y_tr)

        # Hyper-parameter tuning
        print('Searching for optimal LMNN hyper parameters...\n')
        t_bo = time()
        params = {'verbose': 1}
        max_trials = bo.getint('max_trials', fallback=12)
        k_tr, k_te, dim_out, max_iter = bayesopt.find_hyperparams(x_tr, y_tr, x_va, y_va, params, max_trials=max_trials)
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

    verbose = cfg['params'].getint('verbose', fallback=1)
    verbose = 0
    clf = lmnn.LargeMarginNearestNeighbor(n_neighbors=k_tr, max_iter=max_iter, n_features_out=dim_out, verbose=verbose)

    # Train full model
    t_train = time()
    print('Training final model...\n')
    clf = clf.fit(x_tr, y_tr)
    t_train = time() - t_train

    # Print some statistics
    print('\nStatistics:\n{}\nLMNN trained in: {:.4f} s'.format('-'*50, t_train))
    print('Number of iterations: {}'.format(clf.details_['nit']))
    print('Number of function calls: {}'.format(clf.details_['funcalls']))
    print('Average time per function call: {:.4f} s'.format(t_train / clf.details_['funcalls']))
    print('Training loss: {:,}'.format(clf.details_['loss']))
    print('Gradient at the minimum: {}\n'.format(clf.details_['grad']))
    termination_reasons = ['convergence', 'too many iterations or function evaluations']
    warnflag = clf.details_['warnflag']
    termination_reason = clf.details_['task'] if warnflag == 2 else termination_reasons[warnflag]
    print('Training terminated because of {}.\n'.format(termination_reason))


    # Test with simple nearest neighbor classifier
    knn_clf = KNeighborsClassifier(n_neighbors=min(clf.n_neighbors_, k_te))
    t_train = time()
    knn_clf = knn_clf.fit(x_tr, y_tr)
    t_train = time() - t_train
    accuracy_knn = knn_clf.score(x_te, y_te)
    print('\nKNN trained in: {:.4f} s'.format(t_train))
    print('kNN accuracy on test set of {} points: {:.4f}'.format(x_te.shape[0], accuracy_knn))

    # Test with LMNN
    accuracy_lmnn = clf.score(x_te, y_te)
    print('LMNN accuracy on test set of {} points: {:.4f}'.format(x_te.shape[0], accuracy_lmnn))

    # Draw the test data before and after the linear transformation
    plots.plot_comparison(clf.L_, x_te, y_te, dim_pref=3)
    plt.show()


if __name__ == '__main__':
    main(str(sys.argv[1])) if len(sys.argv) > 1 else main()

