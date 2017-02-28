import numpy as np
from time import time
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from pylmnn.bayesopt import findLMNNparams
from pylmnn.lmnn import LMNN
from pylmnn.helpers import test_knn, plot_ba


def load_shape_data(data_dir='shrec14_data', real=False, desc='csd'):
    from os.path import join
    import scipy.io as sio
    f = 'desc_shrec14_real.mat' if real else 'desc_shrec14_synth.mat'
    mat_dict = sio.loadmat(join(data_dir, f))

    if desc == 'sihks':
        data = mat_dict['SihksPooledMat']
    elif desc == 'wks':
        data = mat_dict['WksPooledMat']
    elif desc == 'csd':
        data = np.hstack((mat_dict['WksPooledMat'], mat_dict['SihksPooledMat']))
    else:
        raise TypeError('No descriptor named %s is known.' % desc)

    labels = mat_dict['C']
    # Make labels start from 0 instead of 1
    labels -= 1 if min(labels) == 1 else None
    return data, labels


def main(autotune=True):
    real_or_synth = True
    X, Y = load_shape_data(real=real_or_synth, desc='csd')
    Y = np.array(Y.T[0]).T
    n, d = X.shape
    X = prep.normalize(X)
    print('%d shape descriptors of dimension %d' % (n, d))
    print('Shape labels have dimensions %s' % Y.shape)

    # Separate in training and testing set
    xtr, xte, ytr, yte = train_test_split(X, Y, train_size=0.4, stratify=Y)
    xtr, xva, ytr, yva = train_test_split(xtr, ytr, test_size=0.25, stratify=ytr)

    if autotune:
        # LMNN hyper-parameter tuning
        print('Searching for optimal LMNN hyper parameters...\n')
        t_lmnnParams = time()
        Klmnn, Knn, outdim, maxiter = findLMNNparams(xtr, ytr, xva, yva)
        t_bo = time() - t_lmnnParams
        print('Found optimal LMNN params for %d points in %s\n' % (len(ytr), t_bo))
    else:
        Klmnn, Knn, outdim, maxiter = 2, 1, 44, 198

    # Reconstruct full training set
    xtr = np.concatenate((xtr, xva))
    ytr = np.concatenate((ytr, yva))
    lmnn = LMNN(verbose=True, k=Klmnn, max_iter=maxiter, outdim=outdim)

    t1 = time()
    # Train full model
    print('Training final model...\n')
    lmnn, loss, det = lmnn.fit(xtr, ytr)
    print('LMNN trained in {:.8f}s'.format(time()-t1))

    test_knn(xtr, ytr, xte, yte, k=min(Knn, lmnn.params['k']))
    test_knn(xtr, ytr, xte, yte, k=Knn, L=lmnn.L)
    plot_ba(lmnn.L, xte, yte)


if __name__ == '__main__':
    main(False)


