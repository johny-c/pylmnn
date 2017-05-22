import sys
from time import time
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from .data_fetch import fetch_data
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors.lmnn import LargeMarginNearestNeighbor as LMNN


def benchmark_single(X_train, y_train, X_test, y_test, lmnn_params, dataset):

    knn_clf = KNN(n_neighbors=lmnn_params.get('n_neighbors', 1))
    tic_knn = time()
    knn_clf.fit(X_train, y_train)
    t_knn = time() - tic_knn

    knn_err = 1. - knn_clf .score(X_test, y_test)
    print('KNN Test error on {}: {:5.2f}%'.format(dataset, knn_err*100))

    clf = LMNN(verbose=1, **lmnn_params)
    t_lmnn = time()
    clf.fit(X_train, y_train)
    t_lmnn = time() - t_lmnn
    lmnn_err = 1. - clf.score(X_test, y_test)
    print('KNN  Test error on {}: {:5.2f}%'.format(dataset, knn_err * 100))
    print('LMNN Test error on {}: {:5.2f}%'.format(dataset, lmnn_err*100))

    stats = [knn_err, t_knn, lmnn_err, t_lmnn]

    return stats


def benchmark(dataset):

    with open('lmnn_params.yml', 'r') as config_file:
        params = yaml.load(config_file)

    datasets = params.keys()

    if dataset not in datasets:
        raise NotImplementedError('Currently supported:\n{}'.format(datasets))

    dataset_params = params[dataset]
    n_splits = dataset_params.get('n_splits', 1)
    lmnn_params = dataset_params['lmnn_params']

    dataset_stats = []
    if n_splits == 1:
        X_train, y_train, X_test, y_test = fetch_data(dataset)
        stats = benchmark_single(X_train, y_train, X_test, y_test, lmnn_params,
                         dataset)
        dataset_stats.append(stats)
    else:
        X, y = fetch_data(dataset, split=False)
        for i in range(n_splits):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, stratify=y, random_state=i)
            stats_split = benchmark_single(X_train, y_train, X_test, y_test,
                                  lmnn_params, dataset)

            dataset_stats.append(stats_split)


def main(argv):
    # usage python3 lmnn_demo_usps.py isolet

    dataset = argv[1] if len(argv) > 1 else 'isolet'

    with open('lmnn_params.yml', 'r') as config_file:
        params = yaml.load(config_file)

    benchmark_stats = []
    datasets = params.keys()
    for dataset in params.keys():
        dataset_stats = benchmark(dataset)
        if len(dataset_stats) > 0:
            # compute mean and std
            stats = np.asarray(dataset_stats)
            stats_mean = np.mean(stats, axis=0)
            stats_std  = np.std(stats, axis=0)
        else:
            stats_mean = np.asarray(dataset_stats)
            stats_std = np.zeros_like(stats_mean)

        n_splits = len(dataset_stats)
        stats_mean_std = []
        row_stats = [dataset, n_splits]
        for sm, ss in zip(stats_mean, stats_std):
            smf = '{:5.2f}'.format(sm)
            ssf = '{:5.2f}'.format(ss)
            stats_mean_std = str(smf) + ' (' + str(ssf) + ')'
            row_stats.append(stats_mean_std)


if __name__ == '__main__':
    main(sys.argv)