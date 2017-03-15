import numpy as np
from scipy import sparse, optimize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import gen_batches
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y, check_random_state
import logging, sys, os

from .helpers import unique_pairs, pairs_distances_batch, sum_outer_products, pca_fit


class LargeMarginNearestNeighbor(KNeighborsClassifier):
    """Large Margin Nearest Neighbor metric learning.
    
    This implementation follows closely Kilian Weinberger's MATLAB code found at
    https://bitbucket.org/mlcircus/lmnn which solves the unconstrained problem, finding a linear
    transformation with L-BFGS instead of solving the constrained problem that finds the globally
    optimal metric.
    
    Copyright (c) 2017, John Chiotellis
    Licensed under the GPLv3 license (see LICENSE.txt)

    Parameters
    ----------
    L_init : array_like
        Initial transformation in an array with shape (n_features_out, n_features_in).  If None `load`
        will be used to load a transformation from a file. (default: None)
    n_neighbors : int
        Number of target neighbors (default: 3)
    max_iter : int
        Maximum number of iterations in the optimization (default: 200)
    use_pca : bool
        Whether to use pca to warm-start the linear transformation.
        If False, the identity will be used. (default: True)
    tol : float
        Tolerance for the optimization  (default: 1e-5)
    n_features_out : int
        Preferred dimensionality of the inputs after the transformation.
        If None it is inferred from `use_pca` and `L`.(default: None)
    max_constr : int
        Maximum number of constraints to enforce per iteration (default: 10 million).
    use_sparse : bool
        Whether to use a sparse or a dense matrix for the impostor-pairs storage. Using a sparse matrix,
        the distance to impostors is computed twice, but it is somewhat faster for
        larger data sets than using a dense matrix. With a dense matrix, the unique impostor pairs have to be identified
        explicitly (default: True).
    load : string
        A file path from which to load a linear transformation.
        If None, either identity or pca will be used based on `use_pca` (default: None).
    save : string
        A file path prefix to save intermediate linear transformations to. After every function
        call, it will be extended with the function call number and the `.npy` file
        extension. If None, nothing will be saved (default: None).
    verbose : int
        The level of logger verbosity. Can take values from 0 to 4 inclusive (default: 1).
        0: Only basic information will be printed.
        1: Information from the classifier will be logged.
        2: Information from the classifier and debugging information will be logged.
        3: Information from the classifier and the L-BFGS optimizer will be logged.
        4: Information from the classifier, the L-BFGS optimizer and debugging information will be logged.
    random_state : int
        A seed for reproducibility of random state  (default: None).

    Attributes
    ----------
    X_ : array_like
        An array of training samples with shape (n_samples, n_features_in).
    y_ : array_like
        An array of training labels with shape (n_samples,).
    n_neighbors_ : int
        The number of target neighbors (decreased if n_neighbors was not realistic for all classes)
    classes_: array_like
        An array of the uniquely appearing class labels with shape (n_classes,).
    L_ : array_like
        The linear transformation used during fitting with shape (n_features_out, n_features_in).
    targets_ : array_like
        An array of target neighbors for each sample with shape (n_samples, n_neighbors).
    grad_static_ : array_like
        An array of the gradient component caused by target neighbors, that stays fixed throughout the algorithm with
        shape (n_features_in, n_features_in).
    n_iters_ : int
        The number of iterations of the optimizer.
    n_funcalls_ : int
        The number of times the optimiser computes the loss and the gradient.
    name_ : str
        A name for the instance based on the current number of existing instances.
    logger_ : object
        A logger object to log information during fitting.
    details_ : dict
        A dictionary of information created by the L-BFGS optimizer during fitting.

    _obj_count : int (class attribute)
        An instance counter

    """

    _obj_count = 0

    def __init__(self, L_init=None, n_neighbors=3, n_features_out=None, max_iter=200, tol=1e-5, use_pca=True,
                 max_constr=int(1e7), use_sparse=True, load=None, save=None, verbose=1, random_state=None):

        super().__init__(n_neighbors=n_neighbors)

        # Parameters
        self.L = L_init
        self.n_features_out = n_features_out
        self.max_iter = max_iter
        self.tol = tol
        self.use_pca = use_pca
        self.max_constr = max_constr
        self.use_sparse = use_sparse
        self.load = load
        self.save = save
        self.verbose = verbose
        self.random_state = random_state

        # Initialize number of optimizer iterations and objective function calls
        self.n_iters_ = 0
        self.n_funcalls_ = 0

        # Setup instance name
        LargeMarginNearestNeighbor._obj_count += 1
        self.name_ = __name__ + '(' + str(LargeMarginNearestNeighbor._obj_count) + ')'

    def fit(self, X, y):
        """Find a linear transformation by optimization of the unconstrained problem, such that the k-nearest neighbor
        classification accuracy improves.

        Parameters
        ----------
        X : array_like
            An array of training samples with shape (n_samples, n_features_in).
        y : array_like
            An array of data labels with shape (n_samples,).

        Returns
        -------
        LargeMarginNearestNeighbor
            self

        """

        # Setup logger
        self.logger_ = self._setup_logger()

        # Check inputs consistency
        self.X_, y = check_X_y(X, y)

        # Store the appearing classes and the class index for each sample
        self.classes_, self.y_ = np.unique(y, return_inverse=True)

        # Check that the number of neighbors is achievable for all classes
        self.n_neighbors_ = self.check_n_neighbors()

        # Initialize transformer
        self.L_ = self._init_transformer()

        # Print classifier configuration
        print('Parameters:\n')
        for k, v in self.get_params().items():
            print('{:15}: {}'.format(k, v))
        print()

        # Prepare for saving if needed
        if self.save is not None:
            save_dir, save_file = os.path.split(self.save)
            if save_dir != '' and not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_file = self.save + '_' + str(self.n_funcalls_)
            np.save(save_file, self.L_)

        # Set seed for randomness
        self.random_state_ = check_random_state(self.random_state)

        # Find target neighbors (fixed)
        self.targets_ = self._select_target_neighbors()

        # Compute gradient component of target neighbors (constant)
        self.grad_static_ = self._compute_grad_static()

        # Define optimization problem
        disp = 1 if self.verbose in [3, 4] else None
        self.logger_.info('Now optimizing...')
        L, loss, details = optimize.fmin_l_bfgs_b(func=self._loss_grad, x0=self.L_, bounds=None,
                                                  m=100, pgtol=self.tol, maxfun=500*self.max_iter,
                                                  maxiter=self.max_iter, disp=disp, callback=self._cb)
        # Reshape result from optimizer
        self.L_ = L.reshape(self.n_features_out, L.size // self.n_features_out)

        # Store output to return
        self.details_ = details
        self.details_['loss'] = loss

        # Fit a simple nearest neighbor classifier with the learned metric
        # super().set_params(n_neighbors=self.n_neighbors_)
        super().fit(self.transform(), self.y_)

        return self

    def transform(self, X=None):
        """Applies the learned transformation to the inputs.

        Parameters
        ----------
        X : array_like
            An array of data samples with shape (n_samples, n_features_in) (default: None, defined when fit is called).

        Returns
        -------
        array_like
            An array of transformed data samples with shape (n_samples, n_features_out).

        """
        if X is None:
            X = self.X_
        else:
            X = check_array(X)

        return X @ self.L_.T

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        Returns
        -------
        y_pred : array of shape [n_query]
            Class labels for each data sample.
        """

        # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        y_pred = super().predict(self.transform(X))

        return y_pred

    def _setup_logger(self):
        """Instantiate a logger object for the current class instance"""
        logger = logging.getLogger(self.name_)
        if self.verbose in [1,3]:
            logger.setLevel(logging.INFO)
        elif self.verbose in [2,4]:
            logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(fmt='%(asctime)s  %(name)s - %(levelname)s : %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def check_n_neighbors(self, y=None, n_neighbors=None):
        """Check if all classes have enough samples to query the specified number of neighbors."""

        if y is None:
            y = self.y_

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        max_neighbors = np.bincount(y).min() - 1
        if n_neighbors > max_neighbors:
            self.logger_.warning('n_neighbors too high (={}). Setting to {}\n'.format(n_neighbors, max_neighbors))

        return min(n_neighbors, max_neighbors)

    def _init_transformer(self):
        """Initialise the linear transformation by loading from a file, applying PCA or setting to identity."""
        if self.L is not None:
            return self.L

        if self.load is not None:
            return np.load(self.load)

        if self.use_pca:
            L = pca_fit(self.X_, return_transform=False)
        else:
            L = np.eye(self.X_.shape[1])

        if self.n_features_out is not None:
            n_features_in = self.X_.shape[1]
            if self.n_features_out > n_features_in:
                self.logger_.warning('n_features_out({}) cannot be larger than the inputs dimensionality '
                                    '({}), setting n_features_out to {}!'.format(self.n_features_out, n_features_in, n_features_in))
                self.n_features_out = n_features_in
            L = L[:self.n_features_out]

        return L

    def _select_target_neighbors(self, X=None, n_neighbors=None):
        """Compute the target neighbors, that stay fixed during training.

        Returns
        -------
        array_like
            An array of neighbors indices for each sample with shape (n_samples, n_neighbors).

        """

        if X is None:
            X = self.X_

        if n_neighbors is None:
            n_neighbors = self.n_neighbors_

        self.logger_.info('Finding target neighbors...')
        target_neighbors = np.empty((X.shape[0], n_neighbors), dtype=int)
        for label in self.classes_:
            ind, = np.where(np.equal(self.y_, label))
            dist = euclidean_distances(X[ind], squared=True)
            np.fill_diagonal(dist, np.inf)
            neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
            neigh_ind = neigh_ind[:, :n_neighbors]
            # argpartition doesn't guarantee sorted order, so we sort again but only the k neighbors
            row_ind = np.arange(len(ind))[:, None]
            neigh_ind = neigh_ind[row_ind, np.argsort(dist[row_ind, neigh_ind])]
            target_neighbors[ind] = ind[neigh_ind]

        return target_neighbors

    def _compute_grad_static(self, X=None, target_neighbors=None):
        """Compute the gradient component due to the target neighbors that stays fixed throughout training

        Returns
        -------
        array_like
            An array with the sum of all weighted outer products with shape (n_features_in, n_features_in).

        """

        if X is None:
            X = self.X_

        if target_neighbors is None:
            target_neighbors = self.targets_

        self.logger_.info('Computing gradient component due to target neighbors...')
        n_samples, n_neighbors = target_neighbors.shape
        rows = np.repeat(np.arange(n_samples), n_neighbors)  # 0 0 0 1 1 1 ... (n-1) (n-1) (n-1) with n_neighbors=3
        cols = target_neighbors.flatten()
        targets_sparse = sparse.csr_matrix((np.ones(n_samples * n_neighbors), (rows, cols)), shape=(n_samples, n_samples))

        return sum_outer_products(X, targets_sparse)

    def _cb(self, L):
        """Callback function called after every iteration of the optimizer. The intermediate transformations are
        saved to files if a valid `save` parameter was passed.

        Parameters
        ----------
        L : array_like
            The (flattened) linear transformation in the current iteration.

        """
        self.logger_.info('Iteration {:4} / {:4}'.format(self.n_iters_, self.max_iter))
        if self.save is not None:
            save_file = self.save + '_' + str(self.n_iters_)
            L = L.reshape(self.n_features_out, L.size // self.n_features_out)
            np.save(save_file, L)
        self.n_iters_ += 1

    def _loss_grad(self, L):
        """Compute the loss under a given linear transformation `L` and the loss gradient w.r.t. `L`.

        Parameters
        ----------
        L : array_like
            The current (flattened) linear transformation with shape (n_features_out x n_features_in,).

        Returns
        -------
        tuple
            float: The new loss.
            array_like: The new (flattened) gradient with shape (n_features_out x n_features_in,).

        """

        n_samples, n_features_in = self.X_.shape
        self.L_ = L.reshape(self.n_features_out, n_features_in)
        self.n_funcalls_ += 1
        self.logger_.debug('Function call {}'.format(self.n_funcalls_))

        Lx = self.transform()

        # Compute distances to target neighbors under L (plus margin)
        self.logger_.debug('Computing distances to target neighbors under new L...')
        dist_tn = np.zeros((n_samples, self.n_neighbors_))
        for k in range(self.n_neighbors_):
            dist_tn[:, k] = np.sum(np.square(Lx - Lx[self.targets_[:, k]]), axis=1) + 1

        # Compute distances to impostors under L
        self.logger_.debug('Setting margin radii...')
        margin_radii = np.add(dist_tn[:, -1], 2)

        imp1, imp2, dist_imp = self._find_impostors(Lx, margin_radii, use_sparse=self.use_sparse)

        self.logger_.debug('Computing loss and gradient under new L...')
        loss = 0
        A0 = sparse.csr_matrix((n_samples, n_samples))
        for k in reversed(range(self.n_neighbors_)):
            loss1 = np.maximum(dist_tn[imp1, k] - dist_imp, 0)
            act, = np.where(loss1 != 0)
            A1 = sparse.csr_matrix((2*loss1[act], (imp1[act], imp2[act])), (n_samples, n_samples))

            loss2 = np.maximum(dist_tn[imp2, k] - dist_imp, 0)
            act, = np.where(loss2 != 0)
            A2 = sparse.csr_matrix((2*loss2[act], (imp1[act], imp2[act])), (n_samples, n_samples))

            vals = np.squeeze(np.asarray(A2.sum(0) + A1.sum(1).T))
            A0 = A0 - A1 - A2 + sparse.csr_matrix((vals, (range(n_samples), self.targets_[:, k])), (n_samples, n_samples))
            loss = loss + np.sum(loss1 ** 2) + np.sum(loss2 ** 2)

        grad_new = sum_outer_products(self.X_, A0, remove_zero=True)
        df = self.L_ @ (self.grad_static_ + grad_new)
        df *= 2
        loss = loss + (self.grad_static_ * (self.L_.T @ self.L_)).sum()
        self.logger_.debug('Loss and gradient computed!\n')

        return loss, df.flatten()

    def _find_impostors(self, Lx, margin_radii, use_sparse=True):
        """Compute all impostor pairs exactly.

        Parameters
        ----------
        Lx : array_like
            An array of transformed samples with shape (n_samples, n_features_out).
        margin_radii : array_like
            An array of distances to the farthest target neighbors + margin, with shape (n_samples,).
        use_sparse : bool
            Whether to use a sparse matrix for storing the impostor pairs (default: True).

        Returns
        -------
        tuple: (array_like, array_like, array_like)

            imp1 : array_like
                An array of sample indices with shape (n_impostors,).
            imp2 : array_like
                An array of sample indices that violate a margin with shape (n_impostors,).
            dist : array_like
                An array of pairwise distances of (imp1, imp2) with shape (n_impostors,).

        """
        n_samples = Lx.shape[0]

        self.logger_.debug('Now computing impostor vectors...')
        if use_sparse:
            # Initialize impostors matrix
            impostors_sp = sparse.csr_matrix((n_samples, n_samples), dtype=np.int8)

            for label in self.classes_[:-1]:
                imp1, imp2 = [], []
                idx_in, = np.where(np.equal(self.y_, label))
                idx_out, = np.where(np.greater(self.y_, label))

                # Subdivide idx_out x idx_in to chunks of a size that is fitting in memory
                self.logger_.debug(
                    'Impostor classes {} to class {}..'.format(self.classes_[self.classes_ > label], label))
                ii, jj = self._find_impostors_batch(Lx[idx_out], Lx[idx_in], margin_radii[idx_out],
                                                    margin_radii[idx_in])
                if len(ii):
                    imp1.extend(idx_out[ii])
                    imp2.extend(idx_in[jj])
                    new_imps = sparse.csr_matrix(([1] * len(imp1), (imp1, imp2)), shape=(n_samples, n_samples),
                                                 dtype=np.int8)
                    impostors_sp = impostors_sp + new_imps

            imp1, imp2 = impostors_sp.nonzero()
            # subsample constraints if they are too many
            if impostors_sp.nnz > self.max_constr:
                idx = np.random.choice(impostors_sp.nnz, self.max_constr, replace=False)
                imp1, imp2 = imp1[idx], imp2[idx]

            # self.logger.debug('Computing distances to impostors under new L...')
            dist = pairs_distances_batch(Lx, imp1, imp2)
        else:
            # Initialize impostors vectors
            imp1, imp2, dist = [], [], []
            for label in self.classes_[:-1]:
                idx_in, = np.where(np.equal(self.y_, label))
                idx_out, = np.where(np.greater(self.y_, label))
                # Permute the indices (experimental)
                # idx_in = np.random.permutation(idx_in)
                # idx_out = np.random.permutation(idx_out)

                # Subdivide idx_out x idx_in to chunks of a size that is fitting in memory
                self.logger_.debug(
                    'Impostor classes {} to class {}..'.format(self.classes_[self.classes_ > label], label))
                ii, jj, dd = self._find_impostors_batch(Lx[idx_out], Lx[idx_in], margin_radii[idx_out],
                                                        margin_radii[idx_in], return_dist=True)
                if len(ii):
                    imp1.extend(idx_out[ii])
                    imp2.extend(idx_in[jj])
                    dist.extend(dd)

            idx = unique_pairs(imp1, imp2, n_samples)
            self.logger_.debug('Found {} unique pairs out of {}.'.format(len(idx), len(imp1)))

            # subsample constraints if they are too many
            if len(idx) > self.max_constr:
                # idx = self.random_state.choice(idx, self.max_constr, replace=False)
                idx = np.random.choice(idx, self.max_constr, replace=False)

            imp1 = np.asarray(imp1)[idx]
            imp2 = np.asarray(imp2)[idx]
            dist = np.asarray(dist)[idx]

        return imp1, imp2, dist

    @staticmethod
    def _find_impostors_batch(x1, x2, t1, t2, return_dist=False, batch_size=500):
        """Find impostor pairs in chunks to avoid large memory usage

        Parameters
        ----------
        x1 : array_like
            An array of transformed data samples with shape (n_samples, n_features).
        x2 : array_like
            An array of transformed data samples with shape (m_samples, n_features) where m_samples < n_samples.
        t1 : array_like
            An array of distances to the margins with shape (n_samples,).
        t2 : array_like
            An array of distances to the margins with shape (m_samples,).
        batch_size : int (Default value = 500)
            The size of each chunk of x1 to compute distances to.
        return_dist : bool (Default value = False)
            Whether to return the distances to the impostors.

        Returns
        -------
        tuple: (array_like, array_like, [array_like])
            
            imp1 : array_like
                An array of sample indices with shape (n_impostors,).
            imp2 : array_like
                An array of sample indices that violate a margin with shape (n_impostors,).
            dist : array_like, optional
                An array of pairwise distances of (imp1, imp2) with shape (n_impostors,).

        """

        n, m = len(t1), len(t2)
        imp1, imp2, dist = [], [], []
        for chunk in gen_batches(n, batch_size):
            dist_out_in = euclidean_distances(x1[chunk], x2, squared=True)
            i1, j1 = np.where(dist_out_in < t1[chunk, None])
            i2, j2 = np.where(dist_out_in < t2[None, :])
            if len(i1):
                imp1.extend(i1 + chunk.start)
                imp2.extend(j1)
                if return_dist:
                    dist.extend(dist_out_in[i1, j1])
            if len(i2):
                imp1.extend(i2 + chunk.start)
                imp2.extend(j2)
                if return_dist:
                    dist.extend(dist_out_in[i2, j2])

        if return_dist:
            return imp1, imp2, dist
        else:
            return imp1, imp2
