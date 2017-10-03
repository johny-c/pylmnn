import os
import warnings
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

DPI = 100
COLORS = ["windows blue", "orange red", "grey", "amber"]
PALETTE = sns.xkcd_palette(COLORS)
sns.set(palette=PALETTE)

from config import BENCHMARK_DIR


def plot_time__vs__imp_store(df, save_path=None):

    df['time_per_funcall'] = df['t_fit_lmnn'] / df['n_funcalls']

    # Two subplot columns
    # df_m10 = df.loc[df['max_corrections'] == 10]
    # df_m100 = df.loc[df['max_corrections'] == 100]

    # Set up the matplotlib figure
    # fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig = plt.figure(figsize=(8, 6))
    sns.despine(left=True)

    # Sort the datasets by training set size
    df['x'] = df['dataset'] + '\n(' + df['n_train_samples'].astype(str) + ')'
    order = df[['x', 'n_train_samples']].drop_duplicates()
    order = order.sort_values(by='n_train_samples')['x']
    # order = order.sort_values(by='n_train_samples')['dataset']
    # Sort the colors so the subplots are color consistent
    hue_order = df['imp_store'].unique()

    # Plot time_per_funcall against imp_store
    # ax = axes[0]
    ax = plt.gca()


    data = df  # df_m10
    sns.barplot(x='x', order=order, y='time_per_funcall', data=data,
                hue='imp_store', hue_order=hue_order, ax=ax)
    ax.set_yscale('log')
    # ax.set_title('max_corrections = 10')

    # ax = axes[1]
    # splot100 = sns.barplot(x='dataset', order=order,
    #                        y='time_per_funcall', data=df_m100,
    #                        hue='imp_store', hue_order=hue_order, ax=ax)
    # ax.set_yscale('log')
    # ax.set_title('max_corrections = 100')

    df.apply(annotateBars, ax=ax, axis=1)

    fig.suptitle('Time per function call vs imp_store', fontweight='bold')

    # Save figure
    if save_path is not None:
        fig.savefig(save_path, dpi=DPI)


@DeprecationWarning
def plot_metrics__vs__max_corrections(df, save_path=None):

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

    # Plot metric against max_corrections
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
    if save_path is not None:
        fig.savefig(save_path, dpi=DPI)


def plot_knn__vs__lmnn(df, save_path=None):

    # Keep just one option for max_corrections
    # df = df[df['max_corrections'] == 10]
    # Keep just one option for imp_store
    # df = df[df['imp_store'] == 'sparse']

    df['knn_error100'] = df['knn_error'] * 100
    df['lmnn_error100'] = df['lmnn_error'] * 100

    df_knn = df
    df_lmnn = df.copy()

    df_knn['method'] = 'KNN'
    df_knn = df_knn.rename(columns={'t_fit_knn': 't_fit', 't_test_knn':
        't_test', 'knn_error100': 'test_error%'})

    df_lmnn['method'] = 'LMNN'
    df_lmnn = df_lmnn.rename(columns={'t_fit_lmnn': 't_fit', 't_test_lmnn':
        't_test', 'lmnn_error100': 'test_error%'})
    df = df_knn.append(df_lmnn)

    # Need to show t_fit, t_test, error
    metrics = ['t_fit', 't_test', 'test_error%']

    # Two subplot columns (time, accuracy)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    sns.despine(left=True)

    # Sort the datasets by training set size
    order = df[['dataset', 'n_train_samples']].drop_duplicates()
    order = order.sort_values(by='n_train_samples')['dataset']
    # Sort the colors so the subplots are color consistent
    hue_order = ['KNN', 'LMNN']

    # Plot t_fit, t_test, error against method
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(x='dataset', order=order, y=metric, data=df,
                    ax=ax, hue='method', hue_order=hue_order)

        if metric != 'test_error%':
            ax.set_yscale('log')
        else:
            # ax.set_yticks(np.arange(0, 15, 0.1))
            pass

        ax.set_title('{}'.format(metric))

        plot = df.apply(annotateBars, ax=ax, axis=1)

    fig.suptitle('K-Nearest Neighbors vs LMNN', fontweight='bold')

    # Save figure
    if save_path is not None:
        fig.savefig(save_path, dpi=DPI)


# https://stackoverflow.com/questions/39519609/annotate-bars-with-values-on-pandas-on-seaborn-factorplot-bar-plot
def annotateBars(row, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=14, color='gray',
                    rotation=0, xytext=(0, 10), textcoords='offset points')


def gather_all_runs(log_dir, datasets=None):

    if datasets is None:
        from config import DATASETS
        datasets = DATASETS

    df_all = pd.DataFrame()
    for dataset in datasets:
        dataset_dir = os.path.join(log_dir, dataset)
        if not os.path.exists(dataset_dir):
            msg = 'No results found for dataset {} in benchmark log directory ' \
                  '{}'.format(dataset, log_dir)
            warnings.warn(msg)
            continue

        results_files = os.listdir(dataset_dir)
        results_files = [os.path.join(dataset_dir, f) for f in results_files]

        for results_file in results_files:
            df = pd.read_csv(results_file)
            df_all = df_all.append(df, ignore_index=True)

    return df_all


def get_latest_log_dir():
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument('-l', '--log-dir', type=str, default='latest')
    # return parser

    log_dirs = [os.path.join(BENCHMARK_DIR, ld) for ld in
                os.listdir(BENCHMARK_DIR) if ld.startswith('results')]
    latest_log_dir = max(log_dirs, key=os.path.getctime)

    return latest_log_dir




def main():
    import sys

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
        if not os.path.isdir(log_dir):
            raise NotADirectoryError('{} is not a directory!'.format(log_dir))
    else:
        log_dir = get_latest_log_dir()

    df = gather_all_runs(log_dir)
    save_path = os.path.join(log_dir, 'knn__vs__lmnn.png')
    plot_knn__vs__lmnn(df, save_path)
    plt.show()



if __name__ == '__main__':
    # main()

    log_dir_list = 'results--03-10-2017_15-53-25'
    log_dir_sparse = 'results--03-10-2017_15-56-40'


    log_dir_list = os.path.join(BENCHMARK_DIR, log_dir_list)
    log_dir_sparse = os.path.join(BENCHMARK_DIR, log_dir_sparse)

    df_list = gather_all_runs(log_dir_list)
    df_sparse = gather_all_runs(log_dir_sparse)

    df = df_list.append(df_sparse)
    plot_time__vs__imp_store(df)
    plt.show()