import numpy as np
from argparse import Namespace
from sklearn.neighbors import KNeighborsClassifier
from GPyOpt.methods import BayesianOptimization

from .lmnn import LargeMarginNearestNeighbor


def find_hyperparams(X_train, y_train, X_valid, y_valid, params=None, max_bopt_iter=12):
    """Find the best hyperparameters for LMNN using Bayesian Optimisation.

    Parameters
    ----------

    X_train : array_like
           An array of training samples with shape (n_samples, n_features).

    y_train : array_like
           An array of training labels with shape (n_samples,).
        
    X_valid : array_like
           An array of validation samples with shape (m_samples, n_features).
        
    y_valid : array_like
           An array of validation labels with shape (m_samples,).
        
    params : dict
             A dictionary of parameters to be passed to the LargeMarginNearestNeighbor classifier instance.

    max_bopt_iter : int
            Maximum number of parameter configurations to evaluate (Default value = 12).

    Returns
    -------
    tuple:
        (int, int, int, int) The best hyperparameters found (n_neighbors, n_neighbors_predict, n_components, max_iter).

    """

    params = params or {}
    unique_labels, class_sizes = np.unique(y_train, return_counts=True)
    min_class_size = min(class_sizes)

    # Setting parameters for Bayesian Global Optimization
    args = Namespace()
    args.min_neighbors = 1
    args.max_neighbors = int(min(min_class_size - 1, 15))
    args.min_iter = 10
    args.max_iter = 200
    args.min_components = min(X_train.shape[1], 2)
    args.max_components = X_train.shape[1]

    bopt_iter = 0

    def optimize_clf(hyperparams):
        """The actual objective function with packing and unpacking of hyperparameters.

        Parameters
        ----------
        hyperparams : array_like
                 Vector of hyperparameters to evaluate.

        Returns
        -------
        float
            The validation error obtained.

        """

        hyperparams = hyperparams[0]
        n_neighbors = int(round(hyperparams[0]))
        n_neighbors_predict = int(round(hyperparams[1]))
        n_components = int(np.ceil(hyperparams[2]))
        max_iter = int(np.ceil(hyperparams[3]))

        nonlocal bopt_iter
        bopt_iter += 1
        print('Iteration {} of Bayesian Optimisation'.format(bopt_iter))
        print('Trying n_neighbors(lmnn)={}\tn_neighbors(knn)={}\tn_components={}\tmax_iter={} ...\n'
              .format(n_neighbors, n_neighbors_predict, n_components, max_iter))
        lmnn = LargeMarginNearestNeighbor(n_neighbors, max_iter=max_iter, n_components=n_components, **params)
        lmnn.fit(X_train, y_train)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(lmnn.transform(X_train), y_train)

        print('Evaluating the found transformation on validation set of size {}...'.format(len(y_valid)))
        val_err = 1. - clf.score(lmnn.transform(X_valid), y_valid)

        print('Validation error={:2.4f}\n'.format(val_err))
        return val_err

    # Parameters are discrete but treating them as continuous yields better parameters
    domain = [{'name': 'n_neighbors', 'type': 'continuous', 'domain': (args.min_neighbors, args.max_neighbors)},
              {'name': 'n_neighbors_predict', 'type': 'continuous', 'domain': (args.min_neighbors, args.max_neighbors)},
              {'name': 'n_components', 'type': 'continuous', 'domain': (args.min_components, args.max_components)},
              {'name': 'max_iter', 'type': 'continuous', 'domain': (args.min_iter, args.max_iter)}]

    bo = BayesianOptimization(f=optimize_clf, domain=domain)
    bo.run_optimization(max_iter=max_bopt_iter)

    solution = bo.x_opt
    print(solution)
    best_n_neighbors = int(round(solution[0]))
    best_n_neighbors_predict = int(round(solution[1]))
    best_n_components = int(np.ceil(solution[2]))
    best_max_iter = int(np.ceil(solution[3]))
    print('Best parameters: n_neighbors(lmnn)={} n_neighbors(knn)={} n_components={} max_iter={}\n'.
          format(best_n_neighbors, best_n_neighbors_predict, best_n_components, best_max_iter))

    return best_n_neighbors, best_n_neighbors_predict, best_n_components, best_max_iter
