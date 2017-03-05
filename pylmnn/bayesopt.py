import numpy as np
from GPyOpt.methods import BayesianOptimization
from sklearn.neighbors import KNeighborsClassifier
from .lmnn import LargeMarginNearestNeighbor


def find_hyperparams(x_tr, y_tr, x_va, y_va, params, max_trials=12):
    """Find optimal hyperparameters for LMNN using Bayesian Optimization

    Args:
        x_tr:       [N, D] array-like, training inputs
        y_tr:       [N,]   array_like, training labels
        x_va:       [M, D] array-like, validation inputs
        y_va:       [M,]   array_like, validation labels
        params:     dict, parameters to pass to the LargeMarginNearestNeighbor classifier
        max_trials: maximum number of hyper-parameter configurations to test (Default value = 12)

    Returns:
        k_tr_bo:    int, optimal number of target neighbors during training
        k_te_bo:    int, optimal number of reference neighbors during testing
        dim_out_bo: int, optimal dimensionality for the transformed inputs
        max_iter_bo int, optimal number of iterations for the optimization routine
    """

    unique_labels, class_sizes = np.unique(y_tr, return_counts=True)
    min_class_size = min(class_sizes)

    # Setting parameters for Bayesian Global Optimization
    class BOptions: pass

    opt = BOptions()
    opt.max_trials = max_trials  # How many parameter settings do you want to try?
    opt.min_k = 1
    opt.max_k = int(min(min_class_size - 1, 15))
    opt.min_iter = 10
    opt.max_iter = 200
    opt.min_dim = min(x_tr.shape[1], 2)
    opt.max_dim = x_tr.shape[1]

    bo_iter = 0

    def optimize_clf(hp_vec):
        hp_vec = hp_vec[0]
        k_tr = int(round(hp_vec[0]))
        k_te = int(round(hp_vec[1]))
        dim_out = int(np.ceil(hp_vec[2]))
        max_iter = int(np.ceil(hp_vec[3]))

        nonlocal bo_iter
        bo_iter += 1
        print('Iteration {} of Bayesian Optimisation'.format(bo_iter))
        print('Trying K(lmnn)={} K(knn)={} dim_out={} max_iter={} ...\n'.
              format(k_tr, k_te, dim_out, max_iter))
        lmnn_clf = LargeMarginNearestNeighbor(k=k_tr, max_iter=max_iter, dim_out=dim_out, **params)
        knn_clf = KNeighborsClassifier(n_neighbors=k_te)

        lmnn_clf = lmnn_clf.fit(x_tr, y_tr)
        Lx_tr = lmnn_clf.transform(x_tr)

        knn_clf.fit(Lx_tr, y_tr)
        Lx_va = lmnn_clf.transform(x_va)
        y_pred = knn_clf.predict(Lx_va)
        val_err = np.mean(np.not_equal(y_pred, y_va))

        print('\nvalidation error={:2.4f}\n'.format(val_err))
        return val_err

    # Parameters are discrete but treating them as continuous yields better parameters
    domain = [{'name': 'k_tr', 'type': 'continuous', 'domain': (opt.min_k, opt.max_k)},
              {'name': 'k_te', 'type': 'continuous', 'domain': (opt.min_k, opt.max_k)},
              {'name': 'dim_out', 'type': 'continuous', 'domain': (opt.min_dim, opt.max_dim)},
              {'name': 'max_iter', 'type': 'continuous', 'domain': (opt.min_iter, opt.max_iter)}]

    bo = BayesianOptimization(f=lambda x: optimize_clf(x), domain=domain)
    bo.run_optimization(max_iter=opt.max_trials)

    hp = bo.x_opt
    print(hp)
    k_tr_bo = int(round(hp[0]))
    k_te_bo = int(round(hp[1]))
    dim_out_bo = int(np.ceil(hp[2]))
    max_iter_bo = int(np.ceil(hp[3]))
    print('Best parameters: K(lmnn)={} K(knn)={} dim_out={} max_iter={}\n'.
          format(k_tr_bo, k_te_bo, dim_out_bo, max_iter_bo))
    return k_tr_bo, k_te_bo, dim_out_bo, max_iter_bo
