import numpy as np
import numpy.linalg as LA
from scipy import sparse
from sklearn.utils import gen_batches
from sklearn.neighbors import KNeighborsClassifier


def pca_transform(X, var_ratio=1):
    """

    Parameters
    ----------
    X : array_like
        An array of data samples with shape (n_samples, n_features).
    var_ratio : float
        The variance ratio to be captured (Default value = 1).

    Returns
    -------
    array_like
        An array with shape (n_samples, n_components) which is the input samples projected onto `n_components`
        principal components.

    """

    cc = np.cov(X, rowvar=False)  # Mean is removed
    evals, evecs = LA.eigh(cc)  # Get eigenvalues in ascending order, eigenvectors in columns
    evecs = np.fliplr(evecs)  # Flip eigenvectors to get them in descending eigenvalue order

    if var_ratio == 1:
        L = evecs.T
    else:
        evals = np.flip(evals, axis=0)
        var_exp = np.cumsum(evals)
        var_exp = var_exp / var_exp[-1]
        n_components = np.argmax(np.greater_equal(var_exp, var_ratio))
        L = evecs.T[:n_components]  # Set the first n_components eigenvectors as rows of L

    Lx = X.dot(L.T)

    return Lx


def test_knn(x_tr, y_tr, x_te, y_te, n_neighbors, L=None):
    """Compute the k-nearest neighbor accuracy

    Parameters
    ----------
    x_tr : array_like
        An array of training samples with shape (n_samples, n_features).
    y_tr : array_like
        An array of training labels with shape (n_samples,).
    x_te : array_like
        An array of testing samples with shape (m_samples, n_features).
    y_te : array_like
        An array of testing labels with shape (m_samples,) - the ground truth.
    n_neighbors : int
        The number of neighbors to consider.
    L : array_like
        A learned linear transformation (default: None).

    Returns
    -------
    float
        The k-nn accuracy.

    """
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    if L is None:
        knn_clf.fit(x_tr, y_tr)
        y_pred = knn_clf.predict(x_te)
        acc = np.mean(np.equal(y_pred, y_te))
        print('kNN accuracy on test set of {} points: {:.4f}'.format(x_te.shape[0], acc))
    else:
        knn_clf.fit(x_tr.dot(L.T), y_tr)
        y_pred = knn_clf.predict(x_te.dot(L.T))
        acc = np.mean(np.equal(y_pred, y_te))
        print('LMNN accuracy on test set of {} points: {:.4f}\n'.format(x_te.shape[0], acc))
    return acc


def sum_outer_products(X, weights, remove_zero=False):
    """Computes the sum of weighted outer products using a sparse weights matrix

    Parameters
    ----------
    X : array_like
        An array of data samples with shape (n_samples, n_features_in).
    weights : csr_matrix
        A sparse weights matrix (indicating target neighbors) with shape (n_samples, n_samples).
    remove_zero : bool
        Whether to remove rows and columns of the symmetrized weights matrix that are zero (default: False).

    Returns
    -------
    array_like
        An array with the sum of all weighted outer products with shape (n_features_in, n_features_in).

    """
    weights_sym = weights + weights.T
    if remove_zero:
        _, cols = weights_sym.nonzero()
        idx = np.unique(cols)
        weights_sym = weights_sym.tocsc()[:, idx].tocsr()[idx, :]
        X = X[idx]

    n = weights_sym.shape[0]
    diag = sparse.spdiags(weights_sym.sum(axis=0), 0, n, n)
    laplacian = diag.tocsr() - weights_sym
    sodw = X.T @ laplacian @ X
    return sodw


def pairs_distances_batch(X, ind_a, ind_b, batch_size=500):
    """Equivalent to  np.sum(np.square(x[ind_a] - x[ind_b]), axis=1)

    Parameters
    ----------
    X : array_like
        An array of data samples with shape (n_samples, n_features_in).
    ind_a : array_like
        An array of samples indices with shape (m,).
    ind_b : array_like
        Another array of samples indices with shape (m,).
    batch_size :
        Size of each chunk of X to compute distances for (default: 500)

    Returns
    -------
    array-like
        An array of pairwise distances with shape (m,).

    """
    n = len(ind_a)
    res = np.zeros(n)
    for chunk in gen_batches(n, batch_size):
        res[chunk] = np.sum(np.square(X[ind_a[chunk]] - X[ind_b[chunk]]), axis=1)
    return res


def unique_pairs(ind_a, ind_b, n_samples=None):
    """Find the unique pairs contained in zip(ind_a, ind_b)

    Parameters
    ----------
    ind_a : list
        A list with indices of reference samples of length m.
    ind_b : list
        A list with indices of impostor samples of length m.
    n_samples : int, optional
        The total number of samples (= maximum sample index + 1). If None it will be inferred from the indices.

    Returns
    -------
    array-like
         An array of indices of unique pairs with shape (k,) where k <= m.

    """
    # First generate a hash array
    if n_samples is None:
        n_samples = max(np.max(ind_a), np.max(ind_b))
    h = np.array([i * n_samples + j for i, j in zip(ind_a, ind_b)], dtype=np.uint32)

    # Get the indices of the unique elements in the hash array
    _, ind_u = np.unique(h, return_index=True)
    return ind_u