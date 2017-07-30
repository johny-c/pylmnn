from __future__ import print_function

import os
import sys
from time import time
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import data_fetch
from deskewing import deskew
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors.lmnn import LargeMarginNearestNeighbor as LMNN
from tabulate import tabulate


CWD = os.path.split(__file__)[0]
CONFIG_FILE = os.path.join(os.path.join(CWD, 'dataset_params.yml'))
METRICS = ['knn error', 'knn runtime', 'LMNN error', 'LMNN runtime']


def single_run(X_train, y_train, X_test, y_test, lmnn_params, dataset, rng):

    knn_clf = KNN(n_neighbors=lmnn_params.get('n_neighbors', 1), n_jobs=-1)
    tic_knn = time()
    knn_clf.fit(X_train, y_train)
    t_knn = time() - tic_knn
    print('KNN fitted in {:7.2f}s'.format(t_knn))

    t_knn = time()
    knn_err = 1. - knn_clf .score(X_test, y_test)
    t_knn = time() - t_knn
    print('KNN Test error on {}: {:5.2f}% in {:7.2f}s'.format(dataset,
                                                          knn_err*100, t_knn))

    lmnn = LMNN(verbose=1, random_state=rng, **lmnn_params)
    t_lmnn = time()
    lmnn.fit(X_train, y_train)
    t_lmnn = time() - t_lmnn

    knn_clf.fit(lmnn.transform(X_train), y_train)
    lmnn_err = 1. - knn_clf.score(lmnn.transform(X_test), y_test)
    print('KNN  Test error on {}: {:5.2f}%'.format(dataset, knn_err * 100))
    print('LMNN Test error on {}: {:5.2f}%'.format(dataset, lmnn_err*100))

    stats = [knn_err, t_knn, lmnn_err, t_lmnn]

    return stats


def benchmark_single(dataset):

    with open(CONFIG_FILE, 'r') as config_file:
        params = yaml.load(config_file)

    datasets = params.keys()
    if dataset not in datasets:
        raise NotImplementedError('{} is not supported.\n'
                                  'Currently supported datasets:\n{}'
                                  .format(dataset, datasets))

    dataset_params = params[dataset]
    n_splits = dataset_params.get('n_splits', 1)
    lmnn_params = dataset_params['lmnn_params']

    dataset_stats = []
    if n_splits == 1:
        X_train, y_train, X_test, y_test = data_fetch.fetch_data(dataset)

        if dataset_params.get('deskew', False):
            print('Deskewing data set... ', end='', flush=True)
            t = time()
            d = int(np.sqrt(X_train.shape[1]))
            for i in range(len(X_train)):
                X_train[i] = deskew(X_train[i].reshape(d, d)).ravel()

            for i in range(len(X_test)):
                X_test[i] = deskew(X_test[i].reshape(d, d)).ravel()

            print('done in {:8.2f}s'.format(time() - t))

        if dataset_params.get('pca', False):
            print('Computing principal components... ', end='', flush=True)
            t = time()
            pca = PCA(n_components=lmnn_params['n_features_out'])
            X_new = pca.fit_transform(np.concatenate((X_train, X_test)))
            print('done in {:7.2f}.'.format(time() - t))
            X_train, X_test = X_new[:len(X_train)], X_new[len(X_train):]

        stats = single_run(X_train, y_train, X_test, y_test, lmnn_params,
                           dataset, 0)
        dataset_stats.append(stats)
    else:
        X, y = data_fetch.fetch_data(dataset, split=False)
        test_size = dataset_params['test_size']

        if dataset_params.get('pca', False):
            print('Computing principal components...')
            n_features_out = lmnn_params['n_features_out']
            pca = PCA(n_components=n_features_out)
            X = pca.fit_transform(X)

        for i in range(n_splits):
            print('Running {} split {}'.format(dataset, i+1))
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, stratify=y, random_state=i,
                                 test_size=test_size)
            stats_split = single_run(X_train, y_train, X_test, y_test,
                                     lmnn_params, dataset, i)
            dataset_stats.append(stats_split)


    stats = np.asarray(dataset_stats)
    stats_mean = np.mean(stats, axis=0)
    stats_std  = np.std(stats, axis=0)

    stats_mean = {METRICS[i]: stats_mean[i] for i in range(len(METRICS))}
    stats_std = {METRICS[i]: stats_std[i] for i in range(len(METRICS))}

    return n_splits, stats_mean, stats_std


def print_results(benchmark_stats):

    headers = ['Dataset', 'n_splits'] + METRICS
    table = []
    for dataset, (n_splits, stats_mean, stats_std) in benchmark_stats.items():
        row = []
        row.append(dataset)
        row.append(n_splits)

        for metric in METRICS:
            mean_val = stats_mean[metric]
            std_val  = stats_std[metric]
            print('Metric: {}: {} ({})'.format(metric, mean_val, std_val))
            if 'error' in metric:
                mean_val *= 100
                std_val *= 100
            str = '{:5.2f} ({:5.2f})'.format(mean_val, std_val)
            row.append(str)

        table.append(row)

    print(tabulate(table, headers, tablefmt='pipe'))


def main(argv):
    # usage: python3 lmnn_benchmark.py `dataset_name`
    dataset = argv[1] if len(argv) > 1 else 'iris'

    benchmark_stats = {}
    with open(CONFIG_FILE, 'r') as config_file:
        dataset_params = yaml.load(config_file)

    if dataset == 'all':
        for dataset in dataset_params:
            benchmark_stats[dataset] = benchmark_single(dataset)
            print_results(benchmark_stats)
    else:
        benchmark_stats[dataset] = benchmark_single(dataset)
        print_results(benchmark_stats)


if __name__ == '__main__':
    main(sys.argv)


# | Dataset        |   n_splits | knn error     | knn fittime   | LMNN error   | LMNN fittime    |
# |:---------------|-----------:|--------------:|--------------:|-------------:|----------------:|
# | iris           |        100 |  3.29 ( 2.17) | 0.00 ( 0.00)  | 2.98 ( 2.11) |  0.25 ( 0.03)   |
# | olivetti_faces |         10 | 17.58 ( 3.81) | 0.00 ( 0.00)  | 5.75 ( 1.31) |  2.39 ( 0.32)   |
# | letters        |         10 |  4.54 ( 0.27) | 0.02 ( 0.00)  | 3.65 ( 0.18) | 54.43 ( 4.43)   |
# | usps           |          1 |  6.73 ( 0.00) | 0.01 ( 0.00)  | 5.73 ( 0.00) |  67.47 ( 0.00)  |
# | isolet         |          1 |  8.21 ( 0.00) | 0.04 ( 0.00)  | 3.66 ( 0.00) |  568.48 ( 0.00) |
# | mnist          |          1 |  2.81 ( 0.00) | 0.35 ( 0.00)  | 2.14 ( 0.00) | 4725.09 ( 0.00) |
# | mnistPCA       |          1 |  2.30 ( 0.00) | 156.61 (0.00) | 1.84 ( 0.00) | 4661.65 ( 0.00) |
