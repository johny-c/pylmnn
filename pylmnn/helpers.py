import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as LA
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier


def pca_transform(x, var_ratio=1):
    """

    Args:
        x (array_like):     N inputs of dimension D
        var_ratio (float):  the variance ratio to be captured

    Returns:
        array_like:         N inputs projected onto m principal components

    """

    cc = np.cov(x, rowvar=False)  # Mean is removed
    evals, evecs = LA.eigh(cc)  # Get evals in ascending order, evecs in columns
    evecs = np.fliplr(evecs)  # Flip evecs to get them in descending eigenvalue order

    if var_ratio == 1:
        L = evecs.T
    else:
        evals = np.flip(evals, axis=0)
        var_exp = np.cumsum(evals)
        var_exp = var_exp / var_exp[-1]
        outdim = np.argmax(np.greater_equal(var_exp, var_ratio))
        # Set first dim_out eigenvectors as rows of L
        L = evecs.T[:outdim]
    Lx = x.dot(L.T)
    return Lx


def test_knn(x_tr, y_tr, x_te, y_te, k, L=None):
    """Compute the k-nearest neighbor accuracy

    Args:
        x_tr (array_like):  [N, D] training inputs
        y_tr (array_like):  [N,] training labels
        x_te (array_like):  [M, D] testing inputs
        y_te (array_like):  [M,] testing labels (ground truth)
        k (int):            the number of neighbors to consider
        L (array_like):     the learned linear transformation (default: None)

    Returns:
        float:    the k-nn accuracy

    """
    knn_clf = KNeighborsClassifier(n_neighbors=k)
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


def plot_ba(L, x, y, dim_pref=2, t_sne=False):
    """Draw a scatter plot of points, colored by their labels, before and after applying a learned transformation

    Args:
        L (array_like): [d, D] the learned transformation
        x (array_like): [N, D] inputs
        y (array_like): [N,]  labels
        dim_pref (int): the preferred number of dimensions to plot (default: 2)
        t_sne (bool):   whether to use t-SNE to produce the plot or just use the first two dimensions
                        of the inputs (default: False)

    """
    if dim_pref < 2 or dim_pref > 3:
        print('Preferred plot dimensionality must be 2 or 3, setting to 2!')
        dim_pref = 2

    if t_sne:
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=dim_pref, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(x)
        Lx_tsne = tsne.fit_transform(x.dot(L.T))
        x = x_tsne
        Lx = Lx_tsne
    else:
        Lx = x.dot(L.T)

    fig = plt.figure()
    if x.shape[1] > 2 and dim_pref == 3:
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Lx[:, 0], Lx[:, 1], Lx[:, 2], c=y)
        ax.set_title('Transformed Data')
    elif x.shape[1] >= 2:
        ax = fig.add_subplot(121)
        ax.scatter(x[:, 0], x[:, 1], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122)
        ax.scatter(Lx[:, 0], Lx[:, 1], c=y)
        ax.set_title('Transformed Data')

    plt.show()
