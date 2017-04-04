import os
import numpy as np
from configparser import ConfigParser
from data_fetch import fetch_from_config

from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN


def main(demo='mnist'):
    transf_dir = 'ires_mnist_kw'
    list_of_transf_files = [os.path.join(transf_dir, f) for f in os.listdir(
        transf_dir)]
    latest_trasf_file = max(list_of_transf_files, key=os.path.getctime)
    print('Loading transformation from {}'.format(latest_trasf_file))
    L = np.load(latest_trasf_file)

    if demo not in ['shrec14', 'mnist', 'letters', 'usps', 'isolet', 'faces']:
        raise FileNotFoundError('Unknown dataset {}! Exiting.'.format(demo))

    cfg = ConfigParser()
    cfg.read(demo + '.cfg')
    data_set_name = cfg['fetch']['name']
    print('Data set name: {}'.format(data_set_name))

    X_tr, X_te, y_tr, y_te = fetch_from_config(cfg)

    hyper_params = cfg['hyper_params']
    k_tr = hyper_params.getint('k_tr')
    k_te = hyper_params.getint('k_te')
    dim_out = hyper_params.getint('dim_out')
    max_iter = hyper_params.getint('max_iter')

    clf = LMNN()
    clf.classes_ = np.unique(y_tr)
    clf.L_ = L
    clf.n_neighbors_ = k_tr

    print('Performing PCA on dataset...')
    cov_ = np.cov(np.concatenate((X_tr, X_te)), rowvar=False)  # Mean is
    # removed
    _, evecs = np.linalg.eigh(cov_)
    evecs = np.fliplr(evecs)  # Sort by descending eigenvalues
    # L = evecs.T  # Get as eigenvectors as rows

    evecs = evecs[:, :len(L.T)]
    X_tr_ = X_tr.dot(evecs)
    X_te_ = X_te.dot(evecs)

    print('Performing energy-based classification...')
    y_pred = clf.predict_energy(X_tr_, y_tr, X_te_)


    acc_energy = np.mean(np.equal(y_pred, y_te))
    print('Energy accuracy: {}'.format(acc_energy))

if __name__ == '__main__':
    main()