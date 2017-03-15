import numpy as np
from GPyOpt.methods import BayesianOptimization
from .lmnn import LargeMarginNearestNeighbor


def find_hyperparams(x_tr, y_tr, x_va, y_va, params, max_trials=12):
    """Find the best hyperparameters for LMNN for a specified number of trials using Bayesian Optimisation.

    Parameters
    ----------

    x_tr : array_like
           An array of training samples with shape (n_samples, n_features).

    y_tr : array_like
           An array of training labels with shape (n_samples,).
        
    x_va : array_like
           An array of validation samples with shape (m_samples, n_features).
        
    y_va : array_like
           An array of validation labels with shape (m_samples,).
        
    params : dict
             A dictionary of parameters to be passed to the LargeMarginNearestNeighbor classifier instance.

    max_trials : int
            Maximum number of parameter vectors to evaluate (Default value = 12).

    Returns
    -------
    tuple:
        (int, int, int, int) The best hyperparameters found (k_tr, k_te, n_features_out, max_iter).

    """

    unique_labels, class_sizes = np.unique(y_tr, return_counts=True)
    min_class_size = min(class_sizes)

    # Setting parameters for Bayesian Global Optimization
    class BOptions: pass

    opt = BOptions()
    opt.max_trials = max_trials
    opt.min_k = 1
    opt.max_k = int(min(min_class_size - 1, 15))
    opt.min_iter = 10
    opt.max_iter = 200
    opt.min_dim = min(x_tr.shape[1], 2)
    opt.max_dim = x_tr.shape[1]

    bo_iter = 0

    def optimize_clf(hp_vec):
        """The actual objective function with packing and unpacking of hyperparameters.

        Parameters
        ----------
        hp_vec : array_like
                 Vector of hyperparameters to evaluate.

        Returns
        -------
        float
            The validation error obtained.

        """

        hp_vec = hp_vec[0]
        k_tr = int(round(hp_vec[0]))
        k_te = int(round(hp_vec[1]))
        dim_out = int(np.ceil(hp_vec[2]))
        max_iter = int(np.ceil(hp_vec[3]))

        nonlocal bo_iter
        bo_iter += 1
        print('Iteration {} of Bayesian Optimisation'.format(bo_iter))
        print('Trying K(lmnn)={}\tK(knn)={}\tdim_out={}\tmax_iter={} ...\n'.format(k_tr, k_te, dim_out, max_iter))
        clf = LargeMarginNearestNeighbor(n_neighbors=k_tr, max_iter=max_iter, n_features_out=dim_out, **params)
        clf = clf.fit(x_tr, y_tr)

        print('Evaluating the found transformation on validation set of size {}...'.format(len(y_va)))
        val_err = 1. - clf.score(x_va, y_va)

        print('\nValidation error={:2.4f}\n'.format(val_err))
        return val_err

    # Parameters are discrete but treating them as continuous yields better parameters
    domain = [{'name': 'k_tr', 'type': 'continuous', 'domain': (opt.min_k, opt.max_k)},
              {'name': 'k_te', 'type': 'continuous', 'domain': (opt.min_k, opt.max_k)},
              {'name': 'dim_out', 'type': 'continuous', 'domain': (opt.min_dim, opt.max_dim)},
              {'name': 'max_iter', 'type': 'continuous', 'domain': (opt.min_iter, opt.max_iter)}]

    bo = BayesianOptimization(f=optimize_clf, domain=domain)
    bo.run_optimization(max_iter=opt.max_trials)

    hp = bo.x_opt
    print(hp)
    k_tr_bo = int(round(hp[0]))
    k_te_bo = int(round(hp[1]))
    dim_out_bo = int(np.ceil(hp[2]))
    max_iter_bo = int(np.ceil(hp[3]))
    print('Best parameters: K(lmnn)={} K(knn)={} n_features_out={} max_iter={}\n'.
          format(k_tr_bo, k_te_bo, dim_out_bo, max_iter_bo))

    return k_tr_bo, k_te_bo, dim_out_bo, max_iter_bo
