import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for projection='3d'
import numpy as np
import numpy.linalg as LA
from sklearn import manifold
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


def plot_comparison(L, X, y, dim_pref=2, t_sne=False):
    """Draw a scatter plot of points, colored by their labels, before and after applying a learned transformation

    Parameters
    ----------
    L : array_like
        The learned transformation in an array with shape (n_features_out, n_features_in).
    X : array_like
        An array of data samples with shape (n_samples, n_features_in).
    y : array_like
        An array of data labels with shape (n_samples,).
    dim_pref : int
        The preferred number of dimensions to plot (default: 2).
    t_sne : bool
        Whether to use t-SNE to produce the plot or just use the first two dimensions
        of the inputs (default: False).

    """
    if dim_pref < 2 or dim_pref > 3:
        print('Preferred plot dimensionality must be 2 or 3, setting to 2!')
        dim_pref = 2

    if t_sne:
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=dim_pref, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        Lx_tsne = tsne.fit_transform(X.dot(L.T))
        X = X_tsne
        Lx = Lx_tsne
    else:
        Lx = X.dot(L.T)

    fig = plt.figure()
    if X.shape[1] > 2 and dim_pref == 3:
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Lx[:, 0], Lx[:, 1], Lx[:, 2], c=y)
        ax.set_title('Transformed Data')
    elif X.shape[1] >= 2:
        ax = fig.add_subplot(121)
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122)
        ax.scatter(Lx[:, 0], Lx[:, 1], c=y)
        ax.set_title('Transformed Data')

    plt.show()
