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

from matplotlib import pyplot as plt
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



def gather_all_runs(datasets=DATASETS):

    df_all = pd.DataFrame()
    for dataset in datasets:
        dataset_dir = os.path.join(LOG_DIR, dataset)
        if not os.path.exists(dataset_dir):
            raise NotADirectoryError

        results_files = os.listdir(dataset_dir)
        results_files = [os.path.join(dataset_dir, f) for f in results_files]

        for results_file in results_files:
            df = pd.read_csv(results_file)
            df_all = df_all.append(df, ignore_index=True)

    return df_all


def plot_time_vs_imp_store():

    df = gather_all_runs()
    df['time_per_funcall'] = df['t_fit_lmnn'] / df['n_funcalls']

    # Two subplot columns
    df_m10 = df.loc[df['max_corrections'] == 10]
    df_m100 = df.loc[df['max_corrections'] == 100]

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.despine(left=True)

    # Sort the datasets by training set size
    order = df[['dataset', 'n_train_samples']].drop_duplicates()
    order = order.sort_values(by='n_train_samples')['dataset']
    # Sort the colors so the subplots are color consistent
    hue_order = df['imp_store'].unique()

    # Plot time_per_funcall against imp_store
    ax = axes[0]
    splot10 = sns.barplot(x='dataset', order=order,
                          y='time_per_funcall', data=df_m10,
                          hue='imp_store', hue_order=hue_order, ax=ax)
    ax.set_yscale('log')
    ax.set_title('max_corrections = 10')

    ax = axes[1]
    splot100 = sns.barplot(x='dataset', order=order,
                           y='time_per_funcall', data=df_m100,
                           hue='imp_store', hue_order=hue_order, ax=ax)
    ax.set_yscale('log')
    ax.set_title('max_corrections = 100')

    fig.suptitle('Time per function call vs imp_store', fontweight='bold')

    # Save figure
    save_path = os.path.join(LOG_DIR, 'time__vs__imp_store.png')
    fig.savefig(save_path, dpi=250)


def plot_metrics_vs_max_corrections():

    df = gather_all_runs()
    df['lmnn_error100'] = df['lmnn_error'] * 100
    # Need to show lmnn_error, n_iterations, t_fit_lmnn
    metrics = ['lmnn_error100', 'n_iterations', 't_fit_lmnn']

    # Two subplot columns
    df_list = df.loc[df['imp_store'] == 'list']
    df_sparse = df.loc[df['imp_store'] == 'sparse']

    # Set up the matplotlib figure
    fig, axes = plt.subplots(len(metrics), 2, figsize=(15, 18))
    sns.despine(left=True)

    # Sort the datasets by training set size
    order = df[['dataset', 'n_train_samples']].drop_duplicates()
    order = order.sort_values(by='n_train_samples')['dataset']
    # Sort the colors so the subplots are color consistent
    hue_order = df['max_corrections'].unique()

    # Plot time_per_fev againse imp_store for m=10
    for i, metric in enumerate(metrics):
        for j, imp_store in enumerate(['list', 'sparse']):
            ax = axes[i, j]
            data = df_list if imp_store == 'list' else df_sparse
            s = sns.barplot(x='dataset', order=order, y=metric, data=data,
                            hue='max_corrections', hue_order=hue_order, ax=ax)

            if metric == 't_fit_lmnn':
                ax.set_yscale('log')
            ax.set_title('imp_store = {}'.format(imp_store))

    fig.suptitle('Influence of max_corrections', fontweight='bold')

    # Save figure
    save_path = os.path.join(LOG_DIR, 'metrics__vs__max_corrections.png')
    fig.savefig(save_path, dpi=250)


def main():
    run_benchmark(max_corrections=10, imp_store='list')
    run_benchmark(max_corrections=10, imp_store='sparse')
    run_benchmark(max_corrections=100, imp_store='list')
    run_benchmark(max_corrections=100, imp_store='sparse')


if __name__ == '__main__':
    main()