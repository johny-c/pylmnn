import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier


def clean_data(x, var_ratio=1):
    """
    Apply PCA and get the first k dimensions so that the variance fraction given is explained
    :param x:       NxD inputs
    :param var_ratio:    scalar, float, variance ratio to be captured
    :return:        Nxd transformed inputs
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
        # Set first outdim eigenvectors as rows of L
        L = evecs.T[:outdim]
    Lx = x.dot(L.T)
    return Lx


def test_knn(xtr, ytr, xte, yte, k, L=None):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    if L is None:
        knn_clf.fit(xtr, ytr)
        Y_pred = knn_clf.predict(xte)
        acc = np.mean(np.equal(Y_pred, yte))
        print('kNN accuracy on test set of {} points is {:.4f}'.format(xte.shape[0], acc))
    else:
        knn_clf.fit(xtr.dot(L.T), ytr)
        Y_pred = knn_clf.predict(xte.dot(L.T))
        acc = np.mean(np.equal(Y_pred, yte))
        print('LMNN accuracy on test set of {} points is {:.4f}'.format(xte.shape[0], acc))
    return acc


def plot_ba(L, x, y, tsne=False):
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

