import numpy as np
from GPyOpt.methods import BayesianOptimization
from sklearn.neighbors import KNeighborsClassifier

from pylmnn.lmnn import LMNN


def findLMNNparams(xtr, ytr, xva, yva):
    """
    Find optimal hyperparameters using Bayesian Optimization
    :param xtr: NxD training inputs
    :param ytr: Nx1 training labels
    :param xva: MxD validation inputs
    :param yva: MxD validation labels
    :return:    Klmnn, optimal number of target neighbors during training,
                knn,   optimal number of reference neighbors during testing,
                outdim, optimal dimensionality for the transformed inputs,
                maxiter, optimal number of iterations for the optimization routine
    """

    unique_labels, class_sizes = np.unique(ytr, return_counts=True)
    min_class_size = min(class_sizes)

    # Setting parameters for Bayesian Global Optimization
    class BOptions: pass

    opt = BOptions()
    opt.maxtrials = 12  # How many parameter settings do you want to try?
    opt.minK = 1
    opt.maxK = int(min(min_class_size-1, 15))
    opt.minIter = 10
    opt.maxIter = 200
    opt.minDim = min(xtr.shape[1], 2)
    opt.maxDim = xtr.shape[1]

    def optimizeLMNN(hp):
        hp = hp[0]
        K = int(round(hp[0]))
        knn = int(round(hp[1]))
        outdim = int(np.ceil(hp[2]))
        maxiter = int(np.ceil(hp[3]))

        print('Trying K(lmnn)=%i K(knn)=%i outdim=%i maxiter=%i ...\n' % (K, knn, outdim, maxiter))
        lmnn_clf = LMNN(k=K, max_iter=maxiter, verbose=True, outdim=outdim)
        knn_clf  = KNeighborsClassifier(n_neighbors=knn)

        lmnn_clf, _, _ = lmnn_clf.fit(xtr, ytr)
        Lxtr = lmnn_clf.transform(xtr)

        knn_clf.fit(Lxtr, ytr)
        Lxva = lmnn_clf.transform(xva)
        yPred  = knn_clf.predict(Lxva)
        valerr = np.mean(np.not_equal(yPred, yva))

        print('\nvalidation error={:2.4f}\n'.format(valerr))
        return valerr

    f = lambda hp: optimizeLMNN(hp)
    domain = [{'name': 'Klmnn', 'type': 'continuous', 'domain': (opt.minK, opt.maxK)},
              {'name': 'knn', 'type': 'continuous', 'domain': (opt.minK, opt.maxK)},
              {'name': 'outdim', 'type': 'continuous', 'domain': (opt.minDim, opt.maxDim)},
              {'name': 'maxiter', 'type': 'continuous', 'domain': (opt.minIter, opt.maxIter)}]

    bopt = BayesianOptimization(f=f, domain=domain)
    bopt.run_optimization(max_iter=opt.maxtrials)

    hp = bopt.x_opt
    print(hp)
    Klmnn = int(round(hp[0]))
    knn = int(round(hp[1]))
    outdim = int(np.ceil(hp[2]))
    maxiter = int(np.ceil(hp[3]))
    print('Best parameters: K(lmnn)={} K(knn)={} outdim={} maxiter={}\n'.format(Klmnn, knn,
                                                                                outdim, maxiter))
    return Klmnn, knn, outdim, maxiter
