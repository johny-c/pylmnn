import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier


def pca_transform(x, var_ratio=1):
    """Apply PCA and get the first k dimensions so that the variance fraction given is explained

    Args:
        x:          [N, D] array-like, inputs
        var_ratio:  float, variance ratio to be captured

    Returns:        [N, m] array-like, N inputs projected on the m principal components
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
        x_tr:       [N, D] array-like, training inputs
        y_tr:       [N,]   array_like, training labels
        x_tr:       [M, D] array-like, testing inputs
        y_tr:       [M,]   array_like, testing labels (ground-truth)
        L:          [d, D] array-like, if not None, transform the inputs first by applying this
                    transformation (Default value = None)

    Returns:        float, the k-nn accuracy
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


def plot_ba(L, x, y, tsne=False):
    """Draw a scatter plot of points, colored by their labels, before and after applying a
    learned transformation

    Args:
        L:       [d, D] array-like, the learned transformation
        x:       [N, D] array-like, inputs
        y:       [N,]   array_like, labels
        tsne:    bool, whether to use t-SNE to produce the plot or just use the first two
                dimensions of the inputs (Default value = False)

    Returns:        float, the k-nn accuracy
    """
    if tsne:
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(x)
        Lx_tsne = tsne.fit_transform(x.dot(L.T))
        x = x_tsne
        Lx = Lx_tsne
    else:
        Lx = x.dot(L.T)

    if x.shape[1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Lx[:, 0], Lx[:, 1], Lx[:, 2], c=y)
        ax.set_title('Transformed Data')
    elif x.shape[1] == 2:
        plt.subplot(121)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.title('Original Data')
        plt.subplot(122)
        plt.scatter(Lx[:, 0], Lx[:, 1], c=y)
        plt.title('Transformed Data')

    plt.show()

