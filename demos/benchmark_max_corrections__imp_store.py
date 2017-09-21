from __future__ import print_function

import os
import pandas as pd
from datetime import datetime
from time import time
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from dataset_fetcher import fetch_dataset

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import LargeMarginNearestNeighbor as LMNN

import seaborn as sns; sns.set()


CWD = os.path.split(__file__)[0]
CONFIG_FILE = os.path.join(CWD, 'dataset_params.yml')
LOG_DIR = os.path.join(CWD, 'benchmark_results')


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


def single_run(X_train, y_train, X_test, y_test, lmnn_params, dataset):

    knn_clf = KNN(n_neighbors=lmnn_params.get('n_neighbors', 1), n_jobs=-1)
    t_fit_knn = time()
    knn_clf.fit(X_train, y_train)
    t_fit_knn = time() - t_fit_knn
    print('KNN fitted in {:7.2f}s'.format(t_fit_knn))

    t_test_knn = time()
    knn_err = 1. - knn_clf .score(X_test, y_test)
    t_test_knn = time() - t_test_knn
    print('KNN tested in {:7.2f}s'.format(t_test_knn))
    print('KNN Test error on {}: {:5.2f}%'.format(dataset, knn_err*100))

    lmnn = LMNN(**lmnn_params)
    t_fit_lmnn = time()
    lmnn.fit(X_train, y_train)
    knn_clf.fit(lmnn.transform(X_train), y_train)
    t_fit_lmnn = time() - t_fit_lmnn

    t_test_lmnn = time()
    lmnn_err = 1. - knn_clf.score(lmnn.transform(X_test), y_test)
    t_test_lmnn = time() - t_test_lmnn
    print('KNN  Test error on {}: {:5.2f}%'.format(dataset, knn_err * 100))
    print('LMNN Test error on {}: {:5.2f}%'.format(dataset, lmnn_err*100))

    values_to_log = dict(dataset=dataset,
                         n_train_samples=X_train.shape[0],
                         n_test_samples=X_test.shape[0],
                         n_features=X_train.shape[1],
                         n_classes=len(np.unique(y_train)),
                         t_fit_knn=t_fit_knn, t_test_knn=t_test_knn,
                         t_fit_lmnn=t_fit_lmnn, t_test_lmnn=t_test_lmnn,
                         max_corrections=lmnn.max_corrections,
                         imp_store=lmnn.imp_store,
                         n_iterations=lmnn.result_.nit,
                         n_funcalls=lmnn.result_.nfev,
                         objective=lmnn.result_.fun,
                         knn_error=knn_err,
                         lmnn_error=lmnn_err)

    return values_to_log


DATASETS = [
            'iris', 'olivetti_faces', 'letters', 'usps', 'isolet',
            'mnist_deskewed'
            ]


def run_benchmark(datasets=DATASETS, max_corrections=10, imp_store='sparse'):

    extra_params = dict(max_corrections=max_corrections, store_result=True,
                        imp_store=imp_store, random_state=42, verbose=1)

    with open(CONFIG_FILE, 'r') as config_file:
        datasets_params = yaml.load(config_file)

    for dataset in datasets:
        dataset_dir = os.path.join(LOG_DIR, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(dataset_dir, cur_time + '.csv')
        results_log = ResultsLog(path=results_file)
        dataset_params = datasets_params.get(dataset)
        lmnn_params = dataset_params.get('lmnn_params')
        lmnn_params.update(extra_params)

        X_train, y_train, X_test, y_test = fetch_dataset(dataset)
        n_splits = dataset_params.get('n_splits', 1)
        test_size = dataset_params.get('test_size', 0.3)

        if dataset_params.get('pca', False):
            print('Computing principal components...')
            n_features_out = lmnn_params['n_features_out']
            pca = PCA(n_components=n_features_out, random_state=42)
            X_all = X_train if X_test is None else np.concatenate((X_train,
                                                                   X_test))
            X_all = pca.fit_transform(X_all)
            if X_test is None:
                X_train = X_all
            else:
                X_train, X_test = X_all[:len(X_train)], X_all[len(X_train):]

        for i_split in range(n_splits):
            if X_test is None:
                X_tr, X_te, y_tr, y_te = \
                    train_test_split(X_train, y_train, test_size=test_size,
                                     stratify=y_train, random_state=i_split)
            else:
                X_tr, X_te, y_tr, y_te = X_train, X_test, y_train, y_test

            result = single_run(X_tr, y_tr, X_te, y_te, lmnn_params, dataset)
            results_log.append(**result)

        results_log.save()


def plot_time_vs_imp_store(datasets=DATASETS, max_corrections=100):

    xs, ys, stds, cs = [], [], [], []

    for dataset in datasets:
        dataset_dir = os.path.join(LOG_DIR, dataset)
        if not os.path.exists(dataset_dir):
            raise NotADirectoryError

        results_files = os.listdir(dataset_dir)
        results_files = [os.path.join(dataset_dir, f) for f in results_files]

        for results_file in results_files:
            df = pd.read_csv(results_file)
            max_corrections_ = df['max_corrections'][0]
            if max_corrections_ != max_corrections:
                continue

            n_samples = df['n_train_samples'][0]
            imp_store = df['imp_store'][0]
            vals = df['t_fit_lmnn'] / df['n_funcalls']
            xs.append(imp_store)
            ys.append(vals.mean())
            stds.append(vals.std())
            cs.append(dataset + '\n(n=' + str(n_samples) + ')')

    df = pd.DataFrame({'imp_store': xs, 'time_per_fev': ys, 'dataset': cs,
                       'stds': stds})

    sns_plot = sns.barplot(x='dataset', y='time_per_fev', data=df,
                        hue='imp_store')
    sns.plt.title('Time per function call vs `imp_store`', fontweight='bold')
    save_path = os.path.join(LOG_DIR, 'time__vs__imp_store_m={}.png'.format(
        max_corrections))
    sns_plot.get_figure().savefig(save_path, dpi=250)


def plot_time_vs_maxcor(datasets=DATASETS, imp_store='list'):

    xs, ys, stds, cs = [], [], [], []

    for dataset in datasets:
        dataset_dir = os.path.join(LOG_DIR, dataset)
        if not os.path.exists(dataset_dir):
            raise NotADirectoryError

        results_files = os.listdir(dataset_dir)
        results_files = [os.path.join(dataset_dir, f) for f in results_files]

        for results_file in results_files:
            df = pd.read_csv(results_file)
            imp_store_ = df['imp_store'][0]
            if imp_store_ != imp_store:
                continue

            n_samples = df['n_train_samples'][0]
            max_corrections = df['max_corrections'][0]
            vals = df['t_fit_lmnn'] / df['n_iterations']
            xs.append(max_corrections)
            ys.append(vals.mean())
            cs.append(dataset + '\n(n=' + str(n_samples) + ')')

    df = pd.DataFrame({'max_corrections': xs, 'time_per_iter': ys, 'dataset': cs})

    sns_plot = sns.barplot(x='dataset', y='time_per_iter', data=df,
                           hue='max_corrections')
    sns.plt.title('Time per iteration vs `max_corrections`',
                  fontweight='bold')
    save_path = os.path.join(
        LOG_DIR, 'time__vs__max_corrections__imp_store={}.png'.format(imp_store))
    sns_plot.get_figure().savefig(save_path, dpi=250)


def plot_traintime_vs_maxcor(datasets=DATASETS, imp_store='list'):

    xs, ys, stds, cs = [], [], [], []

    for dataset in datasets:
        dataset_dir = os.path.join(LOG_DIR, dataset)
        if not os.path.exists(dataset_dir):
            raise NotADirectoryError

        results_files = os.listdir(dataset_dir)
        results_files = [os.path.join(dataset_dir, f) for f in results_files]

        for results_file in results_files:
            df = pd.read_csv(results_file)
            imp_store_ = df['imp_store'][0]
            if imp_store_ != imp_store:
                continue

            n_samples = df['n_train_samples'][0]
            max_corrections = df['max_corrections'][0]
            vals = df['t_fit_lmnn']
            xs.append(max_corrections)
            ys.append(vals.mean())
            cs.append(dataset + '\n(n=' + str(n_samples) + ')')

    df = pd.DataFrame({'max_corrections': xs, 't_fit': ys, 'dataset': cs})

    sns_plot = sns.barplot(x='dataset', y='t_fit', data=df,
                           hue='max_corrections')
    sns.plt.title('#Training time vs `max_corrections`',
                  fontweight='bold')
    save_path = os.path.join(
        LOG_DIR, 't_fit__vs__max_corrections__imp_store={}.png'.format(
            imp_store))
    sns_plot.get_figure().savefig(save_path, dpi=250)


def main():
    run_benchmark(max_corrections=10, imp_store='list')
    run_benchmark(max_corrections=10, imp_store='sparse')
    run_benchmark(max_corrections=100, imp_store='list')
    run_benchmark(max_corrections=100, imp_store='sparse')


if __name__ == '__main__':
    main()