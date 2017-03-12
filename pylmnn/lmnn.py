import numpy as np
from numpy import linalg as LA
from scipy import sparse, optimize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import gen_batches
import logging, sys, os


class LargeMarginNearestNeighbor:
    """Large Margin Nearest Neighbor metric learning.
    
    This implementation follows closely Kilian Weinberger's MATLAB code found at
    https://bitbucket.org/mlcircus/lmnn
    which solves the unconstrained problem, finding a linear transformation with L-BFGS instead of
    solving the constrained problem that finds the globally optimal metric.
    
    Copyright (c) 2017, John Chiotellis
    Licensed under the GPLv3 license (see LICENSE.txt)

    Parameters
    ----------
    L : array_like
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
    verbose : bool
        Whether to output information from the L-BFGS optimizer. (default: False)
    n_features_out : int
        Preferred dimensionality of the inputs after the transformation.
        If None it is inferred from `use_pca` and `L`.(default: None)
    max_constr : int
        Maximum number of constraints to enforce per iteration. (default: 10 million)
    use_sparse : bool
        Whether to use a sparse or a dense matrix for the impostor-pairs storage. Using a sparse matrix,
        the distance to impostors is computed twice, but it is somewhat faster for
        larger data sets than using a dense matrix. With a dense matrix, the unique impostor pairs have to be identified
        explicitly. (default: True)
    load : string
        A file path from which to load a linear transformation.
        If None, either identity or pca will be used based on `use_pca`. (default: None)
    save : string
        A file path prefix to save intermediate linear transformations to. After every function
        call, it will be extended with the function call number and the `.npy` file
        extension. If None, nothing will be saved. (default: None)
    log_level : int
        The level of logger verbosity (default: logging.INFO)
    random_state : int
        A random state for reproducibility (default: 42)
        Class Attributes:
    obj_count : int
        An instance counter

    Returns
    -------

    """

    obj_count = 0

    def __init__(self, L=None, n_neighbors=3, max_iter=200, use_pca=True, tol=1e-5, verbose=False,
                 n_features_out=None, max_constr=int(1e7), use_sparse=True, load=None, save=None,
                 log_level=logging.INFO, random_state=42):

        # Parameters
        self.L = L
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.use_pca = use_pca
        self.tol = tol
        self.verbose = verbose
        self.n_features_out = n_features_out

        self.max_constr = max_constr
        self.use_sparse = use_sparse
        self.load = load
        self.save = save
        self.random_state = random_state

        # Attributes
        self.targets = None
        self.dfG = None
        self.n_funcalls = 0

        # Setup logger
        LargeMarginNearestNeighbor.obj_count += 1
        if LargeMarginNearestNeighbor.obj_count == 1:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__ + '(' + str(LargeMarginNearestNeighbor.obj_count) + ')')
        self.logger.setLevel(log_level)
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(fmt='%(asctime)s  %(name)s - %(levelname)s : %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def transform(self, X=None):
        """Applies the learned transformation to the inputs

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
            X = self.X
        return X @ self.L.T

    def _check_inputs(self, X, y):
        """Check the input features and labels for consistency

        Parameters
        ----------
        X : array_like
            An array of data samples with shape (n_samples, n_features_in).
            n_features_in
        y : array_like
            An array of data labels with shape (n_samples,).

        Returns
        -------

        """
        assert len(y) == X.shape[0], "Number of labels ({}) does not match the number of " \
                                          "points ({})!".format(len(y), X.shape[0])
        unique_labels, self.label_ind = np.unique(y, return_inverse=True)
        self.labels = np.arange(len(unique_labels))
        max_neighbors = np.bincount(self.label_ind).min() - 1

        if self.n_neighbors > max_neighbors:
            self.logger.warning('K too high. Setting K to {}\n'.format(max_neighbors))
            self.n_neighbors = max_neighbors

        self.X = X

    def _init_transformer(self):
        """Initialise the linear transformation by loading from a file, applying PCA or setting to identity"""
        if self.L is not None:
            return

        if self.load is not None:
            self.L = np.load(self.load)
            return

        if self.use_pca:
            cc = np.cov(self.X, rowvar=False)  # Mean is removed
            evals, evecs = LA.eigh(cc)  # Get eigenvalues in ascending order, eigenvectors in
            # columns
            evecs = np.fliplr(evecs)    # Flip eigenvectors to get them in descending eigenvalue order
            self.L = evecs.T            # Set L rows equal to eigenvectors
        else:
            self.L = np.eye(self.X.shape[1])

        if self.n_features_out is not None:
            n_features_in = self.X.shape[1]
            if self.n_features_out > n_features_in:
                self.logger.warning('n_features_out({}) cannot be larger than the inputs dimensionality '
                                    '({}), setting n_features_out to {}!'.format(self.n_features_out, n_features_in, n_features_in))
                self.n_features_out = n_features_in
            self.L = self.L[:self.n_features_out]

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

        # Check data consistency and fetch_from_config label counts
        self._check_inputs(X, y)

        # Print classifier configuration
        self._print_config()

        # Initialize L
        self._init_transformer()

        # Find target neighbors (fixed)
        self.logger.info('Finding target neighbors...')
        self.targets = self._select_targets()

        # Compute gradient component of target neighbors (constant)
        self.logger.info('Computing gradient component due to target neighbors...')
        n_samples, n_features_in = X.shape
        rows = np.repeat(np.arange(n_samples), self.n_neighbors)  # 0 0 0 1 1 1 2 2 2 ... (n-1) (n-1) (n-1) with n_neighbors=3
        cols = self.targets.flatten()
        target_neighbors = sparse.csr_matrix((np.ones(n_samples * self.n_neighbors), (rows, cols)), shape=(n_samples, n_samples))
        self.dfG = self._sum_outer_products(X, target_neighbors)

        # Define optimization problem
        disp = 1 if self.verbose else None
        self.logger.info('Now optimizing...')
        self.n_funcalls = 0
        if self.save is not None:
            save_dir, save_file = os.path.split(self.save)
            os.mkdir(save_dir) if not os.path.exists(save_dir) else None
            save_file = self.save + '_' + str(self.n_funcalls)
            np.save(save_file, self.L)

        L, loss, details = optimize.fmin_l_bfgs_b(func=self._loss_grad, x0=self.L, bounds=None,
                                                  m=100, pgtol=self.tol, maxfun=500*self.max_iter,
                                                  maxiter=self.max_iter, disp=disp)

        self.L = L.reshape(L.size // n_features_in, n_features_in)
        self.details = details
        self.details['loss'] = loss

        return self

    def _loss_grad(self, L):
        """Compute the loss under a given L and the loss gradient w.r.t. L

        Parameters
        ----------
        L : array_like
            The current (flattened) linear transformation (n_features_out x n_features_in,).

        Returns
        -------
        tuple
            float: The new loss.
            array_like: The new (flattened) gradient (n_features_out x n_features_in,).

        """
        n_samples, n_features_in = self.X.shape
        _, k = self.targets.shape
        self.L = L.reshape(L.size // n_features_in, n_features_in)
        self.n_funcalls += 1
        self.logger.info('Function call {}'.format(self.n_funcalls))
        if self.save is not None:
            save_file = self.save + '_' + str(self.n_funcalls)
            np.save(save_file, self.L)
        Lx = self.transform()

        # Compute distances to target neighbors under L (plus margin)
        self.logger.debug('Computing distances to target neighbors under new L...')
        dist_tn = np.zeros((n_samples, k))
        for j in range(k):
            dist_tn[:, j] = np.sum(np.square(Lx - Lx[self.targets[:, j]]), axis=1) + 1

        # Compute distances to impostors under L
        self.logger.debug('Setting margin radii...')
        margin_radii = np.add(dist_tn[:, -1], 2)

        if self.use_sparse:
            imp1, imp2, dist_imp = self._find_impostors_sp(Lx, margin_radii)
        else:
            imp1, imp2, dist_imp = self._find_impostors(Lx, margin_radii)

        self.logger.debug('Computing loss and gradient under new L...')
        loss = 0
        A0 = sparse.csr_matrix((n_samples, n_samples))
        for nnid in reversed(range(k)):
            loss1 = np.maximum(dist_tn[imp1, nnid] - dist_imp, 0)
            act, = np.where(loss1 != 0)
            A1 = sparse.csr_matrix((2*loss1[act], (imp1[act], imp2[act])), (n_samples, n_samples))

            loss2 = np.maximum(dist_tn[imp2, nnid] - dist_imp, 0)
            act, = np.where(loss2 != 0)
            A2 = sparse.csr_matrix((2*loss2[act], (imp1[act], imp2[act])), (n_samples, n_samples))

            vals = np.squeeze(np.asarray(A2.sum(0) + A1.sum(1).T))
            A0 = A0 - A1 - A2 + sparse.csr_matrix((vals, (range(n_samples), self.targets[:, nnid])), (n_samples, n_samples))
            loss = loss + np.sum(loss1 ** 2) + np.sum(loss2 ** 2)

        sum_outer_prods = self._sum_outer_products(self.X, A0, remove_zero=True)
        df = self.L @ (self.dfG + sum_outer_prods)
        df *= 2
        loss = loss + (self.dfG * (self.L.T @ self.L)).sum()
        self.logger.debug('Loss and gradient computed!\n')
        return loss, df.flatten()

    def _select_targets(self):
        """Compute target neighbors, that stay fixed during training

        Parameters
        ----------

        Returns
        -------
        array_like
            An array of neighbors indices for each sample with shape (n_samples, n_neighbors).

        """

        target_neighbors = np.empty((self.X.shape[0], self.n_neighbors), dtype=int)
        for label in self.labels:
            ind, = np.where(np.equal(self.label_ind, label))
            dist = euclidean_distances(self.X[ind], squared=True)
            np.fill_diagonal(dist, np.inf)
            neigh_ind = np.argpartition(dist, self.n_neighbors - 1, axis=1)
            neigh_ind = neigh_ind[:, :self.n_neighbors]
            # argpartition doesn't guarantee sorted order, so we sort again but only the n_neighbors neighbors
            row_ind = np.arange(len(dist))[:, None]
            neigh_ind = neigh_ind[row_ind, np.argsort(dist[row_ind, neigh_ind])]
            target_neighbors[ind] = ind[neigh_ind]

        return target_neighbors

    def _find_impostors(self, Lx, margin_radii):
        """Compute all impostor pairs exactly

        Parameters
        ----------
        Lx : array_like
            An array of transformed samples with shape (n_samples, n_features_out).
        margin_radii : array_like
            An array of distances to the farthest target neighbors + margin, with shape (n_samples,)

        Returns
        -------
        tuple
            
        array_like
            Impostor pairs and their distances

        """
        n_samples = self.X.shape[0]

        # Initialize impostors vectors
        imp1, imp2, dist = [], [], []
        self.logger.debug('Now computing impostor vectors...')
        for label in self.labels[:-1]:
            idx_in, = np.where(np.equal(self.label_ind, label))
            idx_out, = np.where(np.greater(self.label_ind, label))
            # Permute the indices (experimental)
            # idx_in = np.random.permutation(idx_in)
            # idx_out = np.random.permutation(idx_out)

            # Subdivide idx_out x idx_in to chunks of a size that is fitting in memory
            self.logger.debug('Impostor classes {} to class {}..'.
                          format(self.labels[self.labels > label], label))
            ii, jj, dd = self._find_impostors_batch(Lx[idx_out], Lx[idx_in], margin_radii[idx_out],
                                                    margin_radii[idx_in], return_dist=True)
            if len(ii):
                imp1.extend(idx_out[ii])
                imp2.extend(idx_in[jj])
                dist.extend(dd)

        idx = self._unique_pairs(imp1, imp2, n_samples)

        # subsample constraints if they are too many
        if len(idx) > self.max_constr:
            idx = np.random.choice(idx, self.max_constr, replace=False)

        imp1 = np.asarray(imp1)[idx]
        imp2 = np.asarray(imp2)[idx]
        dist = np.asarray(dist)[idx]
        return imp1, imp2, dist

    def _find_impostors_sp(self, Lx, margin_radii):
        """Compute all impostor pairs exactly using a sparse matrix for storage

        Parameters
        ----------
        Lx : array_like
            An array of transformed samples with shape (n_samples, n_features_out).
        margin_radii : array_like
            An array of distances to the farthest target neighbors + margin, with shape (n_samples,).

        Returns
        -------
        tuple
            
        imp1 : array_like
            (n_impostors,) sample indices
        imp2 : array_like
            (n_impostors,) indices of impostors (samples that violate the margin) to imp1
        dist : array_like
            (n_impostors,) pairwise distances of (imp1, imp2)

        """
        n_samples = self.X.shape[0]

        # Initialize impostors matrix
        impostors_sp = sparse.csr_matrix((n_samples, n_samples), dtype=np.int8)
        self.logger.debug('Now computing impostor vectors...')
        for label in self.labels[:-1]:
            imp1, imp2 = [], []
            idx_in, = np.where(np.equal(self.label_ind, label))
            idx_out, = np.where(np.greater(self.label_ind, label))

            # Subdivide idx_out x idx_in to chunks of a size that is fitting in memory
            self.logger.debug('Impostor classes {} to class {}..'.
                          format(self.labels[self.labels > label], label))
            ii, jj = self._find_impostors_batch(Lx[idx_out], Lx[idx_in], margin_radii[idx_out],
                                                margin_radii[idx_in])
            if len(ii):
                imp1.extend(idx_out[ii])
                imp2.extend(idx_in[jj])
                new_imps = sparse.csr_matrix(([1]*len(imp1), (imp1, imp2)), shape=(n_samples, n_samples), dtype=np.int8)
                impostors_sp = impostors_sp + new_imps

        imp1, imp2 = impostors_sp.nonzero()
        # subsample constraints if they are too many
        if impostors_sp.nnz > self.max_constr:
            idx = np.random.choice(impostors_sp.nnz, self.max_constr, replace=False)
            imp1, imp2 = imp1[idx], imp2[idx]

        # self.logger.debug('Computing distances to impostors under new L...')
        dist = self._cdist(Lx, imp1, imp2)
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
        type
            tuple:
            
            (array_like, array_like) indices of impostor pairs

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

    @staticmethod
    def _sum_outer_products(X, weights, remove_zero=False):
        """Computes the sum of weighted outer products using a sparse weights matrix

        Parameters
        ----------
        X : array_like
            An array of data samples with shape (n_samples, n_features_in).
        weights : csr_matrix
            An sparse matrix indicating target neighbors with shape (n_samples, n_samples).
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

    @staticmethod
    def _cdist(X, ind_a, ind_b, batch_size=500):
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

    def _unique_pairs(self, ind_a, ind_b, n_samples):
        """Find the unique pairs contained in zip(ind_a, ind_b)

        Parameters
        ----------
        ind_a : list
            A list with indices of reference samples of length m.
        ind_b : list
            A list with indices of impostor samples of length m.
        n_samples : int
            The total number of samples (= maximum sample index + 1)

        Returns
        -------
        array-like
             An array of indices of unique pairs with shape (k,) where k <= m.

        """
        # First generate a hash array
        h = np.array([i * n_samples + j for i, j in zip(ind_a, ind_b)])

        # Get the indices of the unique elements in the hash array
        _, ind_u = np.unique(h, return_index=True)
        self.logger.debug('Found {} unique pairs out of {}.'.format(len(ind_u), len(ind_a)))
        return ind_u

    def _print_config(self):
        """Print some parts of the classifier configuration that a user is likely to be interested in. """
        print('Parameters:\n')
        params_to_print = {'n_neighbors', 'n_features_out', 'max_iter', 'use_pca', 'max_constr', 'load', 'save', 'use_sparse', 'tol'}
        for k, v in self.__dict__.items():
            if k in params_to_print:
                print('{:10}: {}'.format(k, v))
        print()
