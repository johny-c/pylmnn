import numpy as np
from sklearn.utils.extmath import row_norms, safe_sparse_dot


def _euclidean_distances_without_checks(X, Y=None, Y_norm_squared=None,
                                        squared=False, X_norm_squared=None,
                                        clip=True):
    """sklearn.pairwise.euclidean_distances without checks with optional clip.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

    Y_norm_squared : array-like, shape (n_samples_2, ), optional
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)

    squared : boolean, optional
        Return squared Euclidean distances.

    X_norm_squared : array-like, shape = [n_samples_1], optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)

    clip : bool, optional (default=True)
        Whether to explicitly enforce computed distances to be non-negative.
        Some algorithms, such as LargeMarginNearestNeighbor, compare distances
        to strictly positive values (distances to farthest target neighbors
        + margin) only to make a binary decision (if a sample is an impostor
        or not). In such cases, it does not matter if the distance is zero
        or negative, since it is definitely smaller than a strictly positive
        value.

    Returns
    -------
    distances : array, shape (n_samples_1, n_samples_2)

    """

    if Y is None:
        Y = X

    if X_norm_squared is not None:
        XX = X_norm_squared
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY

    if clip:
        np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)
