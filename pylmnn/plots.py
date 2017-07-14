import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for projection='3d'
from sklearn import manifold


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

