from __future__ import print_function

import os
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
import yaml

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import LargeMarginNearestNeighbor as LMNN

from config import BENCHMARK_DIR, CONFIG_FILE, DATASETS
from util.dataset_fetcher import fetch_dataset



class ResultsLog:
    def __init__(self, path='results.csv'):
        self.path = path
        self.results = None

    def append(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : a dictionary of names and data to be converted or appended to
        a pandas.DataFrame

        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self):
        self.results.to_csv(self.path, index=False, index_label=False)


def single_run(X_train, y_train, X_test, y_test, lmnn_params, dataset_name):

    knn_clf = KNN(n_neighbors=lmnn_params.get('n_neighbors', 1), n_jobs=-1)
    t_fit_knn = time()
    knn_clf.fit(X_train, y_train)
    t_fit_knn = time() - t_fit_knn
    print('KNN fitted in {:7.2f}s'.format(t_fit_knn))

    t_test_knn = time()
    knn_err = 1. - knn_clf .score(X_test, y_test)
    t_test_knn = time() - t_test_knn
    print('KNN tested in {:7.2f}s'.format(t_test_knn))
    print('KNN Test error on {}: {:5.2f}%'.format(dataset_name, knn_err * 100))

    lmnn = LMNN(**lmnn_params)
    t_fit_lmnn = time()
    lmnn.fit(X_train, y_train)
    knn_clf.fit(lmnn.transform(X_train), y_train)
    t_fit_lmnn = time() - t_fit_lmnn

    t_test_lmnn = time()
    lmnn_err = 1. - knn_clf.score(lmnn.transform(X_test), y_test)
    t_test_lmnn = time() - t_test_lmnn
    print('KNN  Test error on {}: {:5.2f}%'.format(dataset_name, knn_err*100))
    print('LMNN Test error on {}: {:5.2f}%'.format(dataset_name, lmnn_err*100))

    values_to_log = dict(dataset=dataset_name,
                         n_train_samples=X_train.shape[0],
                         n_test_samples=X_test.shape[0],
                         n_features=X_train.shape[1],
                         n_classes=len(np.unique(y_train)),
                         t_fit_knn=t_fit_knn,
                         t_test_knn=t_test_knn,
                         t_fit_lmnn=t_fit_lmnn,
                         t_test_lmnn=t_test_lmnn,
                         imp_store=lmnn.imp_store,
                         n_iterations=lmnn.opt_result_.nit,
                         n_funcalls=lmnn.opt_result_.nfev,
                         objective=lmnn.opt_result_.fun,
                         knn_error=knn_err,
                         lmnn_error=lmnn_err)

    return values_to_log


def run_benchmark(name='', **benchmark_params):

    cur_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    if name != '':
        name += '--'
    log_dir = os.path.join(BENCHMARK_DIR, 'results--' + name + cur_time)

    extra_params = dict(store_opt_result=True, random_state=42, verbose=1)

    with open(CONFIG_FILE, 'r') as config_file:
        datasets_params = yaml.load(config_file)

    for dataset_name in DATASETS:
        dataset_dir = os.path.join(log_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        cur_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        results_file = os.path.join(dataset_dir, cur_time + '.csv')
        results_log = ResultsLog(path=results_file)
        dataset_params = datasets_params.get(dataset_name)
        lmnn_params = dataset_params.get('lmnn_params')
        lmnn_params.update(extra_params)
        lmnn_params.update(**benchmark_params)

        X, y, X_test, y_test = fetch_dataset(dataset_name)
        n_splits = dataset_params.get('n_splits', 1)
        test_size = dataset_params.get('test_size', 0.3)

        pca_flag = dataset_params.get('pca', False)
        n_features_original = X.shape[1]
        if pca_flag:
            print('Computing principal components...')
            n_features_out = lmnn_params['n_features_out']
            pca = PCA(n_components=n_features_out, random_state=42)
            pca.fit(X)
            X = pca.transform(X)
            if X_test is not None:
                X_test = pca.transform(X_test)

        for i_split in range(n_splits):
            if X_test is None:
                X_tr, X_te, y_tr, y_te = \
                    train_test_split(X, y, test_size=test_size, stratify=y,
                                     random_state=i_split)
            else:
                X_tr, X_te, y_tr, y_te = X, X_test, y, y_test

            result = single_run(X_tr, y_tr, X_te, y_te, lmnn_params,
                                dataset_name)
            results_log.append(n_features_original=n_features_original,
                               pca_flag=pca_flag, **result)

        results_log.save()


if __name__ == '__main__':
    run_benchmark(imp_store='sparse')