import numpy as np
from GPyOpt.methods import BayesianOptimization
from sklearn.neighbors import KNeighborsClassifier
from .lmnn import LargeMarginNearestNeighbor


def find_hyper_params(xtr, ytr, xva, yva, params, max_trials=12):
    """
    Find optimal hyperparameters using Bayesian Optimization
    :param xtr: NxD training inputs
    :param ytr: Nx1 training labels
    :param xva: MxD validation inputs
    :param yva: MxD validation labels
    :param max_trials: maximum number of hyper-parameter configurations to evaluate
    :return: Klmnn, optimal number of target neighbors during training, knn, optimal number of
    reference neighbors during testing, dim_out, optimal dimensionality for the transformed
    inputs, maxiter, optimal number of iterations for the optimization routine
    """

    unique_labels, class_sizes = np.unique(ytr, return_counts=True)
    min_class_size = min(class_sizes)

    # Setting parameters for Bayesian Global Optimization
    class BOptions: pass

    opt = BOptions()
    opt.max_trials = max_trials  # How many parameter settings do you want to try?
    opt.min_k = 1
    opt.max_k = int(min(min_class_size - 1, 15))
    opt.min_iter = 10
    opt.max_iter = 200
    opt.min_dim = min(xtr.shape[1], 2)
    opt.max_dim = xtr.shape[1]

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

        lmnn_clf, _, _ = lmnn_clf.fit(xtr, ytr)
        Lxtr = lmnn_clf.transform(xtr)

        knn_clf.fit(Lxtr, ytr)
        Lxva = lmnn_clf.transform(xva)
        y_pred = knn_clf.predict(Lxva)
        val_err = np.mean(np.not_equal(y_pred, yva))

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
    Klmnn = int(round(hp[0]))
    knn = int(round(hp[1]))
    outdim = int(np.ceil(hp[2]))
    maxiter = int(np.ceil(hp[3]))
    print('Best parameters: K(lmnn)={} K(knn)={} dim_out={} max_iter={}\n'.
          format(Klmnn, knn, outdim, maxiter))
    return Klmnn, knn, outdim, maxiter
