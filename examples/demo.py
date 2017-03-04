import numpy as np
from time import time
import sklearn.datasets as skd
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from configparser import ConfigParser

from pylmnn.bayesopt import find_hyper_params
from pylmnn.lmnn import LargeMarginNearestNeighbor
from pylmnn.helpers import test_knn, plot_ba, clean_data

from data_fetch import fetch_load_letters, fetch_load_isolet, load_shrec14


def main(demo='mnist'):

    cfg = ConfigParser()
    cfg.read(demo + '.cfg')

    data_set_name = cfg['fetch']['name']
    if cfg['fetch'].getboolean('sklearn'):
        if data_set_name == 'OLIVETTI':
            data_set = skd.fetch_olivetti_faces(shuffle=True)
        else:
            data_set = skd.fetch_mldata(data_set_name)
        X, y = data_set.data, data_set.target
        if data_set_name == 'MNIST original':
            if cfg['pre_process'].getboolean('normalize'):
                X = X / 255.
    else:
        if data_set_name == 'LETTERS':
            X, y = fetch_load_letters()
        elif data_set_name == 'ISOLET':
            x_tr, x_te, y_tr, y_te = fetch_load_isolet()
        elif data_set_name == 'SHREC14':
            X, y = load_shrec14(real=cfg['fetch']['real'], desc=cfg['fetch']['desc'])
            X = prep.normalize(X, norm=cfg['pre_process']['norm'])
        else:
            raise NameError('No data set {} found!'.format(data_set_name))

    # Separate in training and testing set
    if data_set_name == 'MNIST original':
        x_tr, x_te, y_tr, y_te = X[:60000], X[60000:], y[:60000], y[60000:]
    elif data_set_name != 'ISOLET':
        test_size = cfg['train_test'].getfloat('test_size')
        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y)

    print('{} features of dimension {}'.format(len(y_tr) + len(y_te), x_tr.shape[1]))

    if cfg['pre_process'].getboolean('pca'):
        print('Cleaning data set...')
        X = clean_data(np.concatenate((x_tr, x_te)), var_ratio=0.95)
        x_tr, x_te = X[:x_tr.shape[0]], X[x_tr.shape[0]:]

    bo = cfg['bayes_opt']
    if bo.getboolean('perform'):
        # Separate in training and validation set
        test_size = bo.getfloat('test_size')
        x_tr, xva, y_tr, yva = train_test_split(x_tr, y_tr, test_size=test_size, stratify=y_tr)

        # LMNN hyper-parameter tuning
        print('Searching for optimal LMNN hyper parameters...\n')
        t_bo = time()
        k_tr, k_te, dim_out, max_iter = find_hyper_params(x_tr, y_tr, xva, yva, bo['max_trials'])
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

    clf = LargeMarginNearestNeighbor(verbose=True, k=k_tr, max_iter=max_iter, dim_out=dim_out)

    # Train full model
    t_train = time()
    print('Training final model...\n')
    clf, loss, det = clf.fit(x_tr, y_tr)
    print('LMNN trained in {:.8f}s'.format(time()-t_train))

    test_knn(x_tr, y_tr, x_te, y_te, k=min(k_te, clf.params['k']))
    test_knn(x_tr, y_tr, x_te, y_te, k=k_te, L=clf.L)
    plot_ba(clf.L, x_te, y_te)


if __name__ == '__main__':
    main()
