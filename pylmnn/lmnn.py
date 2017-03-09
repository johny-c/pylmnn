import numpy as np
from numpy import linalg as LA
from scipy import sparse, optimize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import gen_batches
import logging, sys, os


class LargeMarginNearestNeighbor:
    """
    Large Margin Nearest Neighbor metric learning.

    This implementation follows closely Kilian Weinberger's MATLAB code found at
    https://bitbucket.org/mlcircus/lmnn
    which solves the unconstrained problem, finding a linear transformation with L-BFGS instead of
    solving the constrained problem that finds the globally optimal metric.

    Copyright (c) 2017, John Chiotellis
    Licensed under the GPLv3 license (see LICENSE.txt)

    Args:
        L (array_like):     [d, D] initial transformation, if None the load will be used to load a transformation from a
                            file (default: None)
        k (int):            number of target neighbors (default: 3)
        max_iter (int):     maximum number of iterations in the optimization (default: 200)
        use_pca (bool):     whether to use pca to warm-start the linear transformation,
                            if False, the identity will be used (default: True)
        tol (float):        tolerance for the optimization  (default: 1e-5)
        verbose (bool):     whether to output information from the L-BFGS optimizer (default: False)
        dim_out (int):      preferred dimensionality of the inputs after the transformation,
                            if None it is inferred from use_pca and L (default: None)
        max_constr (int):   maximum number of constraints to enforce per iteration (default: 10 million)
        use_sparse (bool):  whether to use a sparse matrix for the impostors storage.
                            Although using this, the distance to impostors is computed twice,
                            this way is somewhat faster for larger data sets than using a dense matrix, where unique
                            pairs have to be identified explicitly. (default: True)
        load (string):      file name from which to load a linear transformation.
                            If None, either identity or pca will be used based on `use_pca`. (default: None)
        save (string):      file name prefix to save intermediate linear transformations to. After every function call,
                            it will be extended with number of function call and `.npy` file extension.
                            If None, nothing is saved. (default: None)
        temp_dir (string):  name of directory to save/load computed transformations to/from (default: 'temp_res')
        log_level (int):    level of logger verbosity (default: logging.INFO)

    Class Attributes:
        obj_count (int):    instance counter

    Attributes:
        targets (array_like):    [N, k], the k target neighbors of each input
        dfG (array_like):        [d, D], the gradient component from target neighbors (computed once)
        n_funcalls (int):        counter of calls to _loss_grad
        logger (object):         logger object, responsible for printing intermediate messages/warnings/etc.
        details (dict):          statistics about the algorithm execution mainly from the L-BFGS optimizer

    """

    obj_count = 0

    def __init__(self, L=None, k=3, max_iter=200, use_pca=True, tol=1e-5, verbose=False,
                 dim_out=None, max_constr=int(1e7), use_sparse=True, load=None, save=None, temp_dir='temp_res',
                 log_level=logging.INFO):

        # Parameters
        self.L = L
        self.k = k
        self.max_iter = max_iter
        self.use_pca = use_pca
        self.tol = tol
        self.verbose = verbose
        self.dim_out = dim_out

        self.max_constr = max_constr
        self.use_sparse = use_sparse
        self.load = load
        self.save = save
        self.temp_dir = temp_dir

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

        Args:
          X (array_like):     [N, D] input samples (default: None, defined when fit is called)

        Returns:
             array_like:     [N, d] transformed inputs

        """
        if X is None:
            X = self.X
        return X @ self.L.T

    def _check_inputs(self, X, y):
        """Check the input features and labels for consistency

        Args:
          X (array_like): [N, D], N input vectors of dimension D
          y (array_like): [N,], N input labels

        """
        assert len(y) == X.shape[0], "Number of labels ({}) does not match the number of " \
                                          "points ({})!".format(len(y), X.shape[0])
        unique_labels, self.label_ind = np.unique(y, return_inverse=True)
        self.labels = np.arange(len(unique_labels))
        max_k = np.bincount(self.label_ind).min() - 1

        if self.k > max_k:
            self.logger.warning('K too high. Setting K to {}\n'.format(max_k))
            self.k = max_k

        self.X = X

    def _init_transformer(self):
        """Initialise the linear transformation by loading from a file, applying PCA or setting to identity """
        if self.L is not None:
            return

        if self.load is not None:
            self.L = np.load(os.path.join(self.temp_dir, self.load))
            return

        if self.use_pca:
            cc = np.cov(self.X, rowvar=False)  # Mean is removed
            evals, evecs = LA.eigh(cc)  # Get evals in ascending order, evecs in columns
            evecs = np.fliplr(evecs)    # Flip evecs to get them in descending eigenvalue order
            self.L = evecs.T            # Set L rows equal to eigenvectors
        else:
            self.L = np.eye(self.X.shape[1])

        if self.dim_out is not None:
            D = self.X.shape[1]
            if self.dim_out > self.L.shape[0]:
                self.logger.warning('dim_out({}) cannot be larger than the inputs dimensionality '
                                    '({}), setting dim_out to {}!'.format(self.dim_out, D, D))
                self.dim_out = D
            self.L = self.L[:self.dim_out]

    def fit(self, X, y):
        """Find a linear transformation by optimization of the unconstrained problem, such that the k-nearest neighbor
        classification accuracy improves

        Args:
          X (array_like): [N, D] training samples
          y (array_like): [N,] class labels of training samples

        Returns:
          LargeMarginNearestNeighbor: self

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
        N, D = X.shape
        rows = np.repeat(np.arange(N), self.k)  # 0 0 0 1 1 1 2 2 2 ... (n-1) (n-1) (n-1) with k=3
        cols = self.targets.flatten()
        target_neighbors = sparse.csr_matrix((np.ones(N*self.k), (rows, cols)), shape=(N, N))
        self.dfG = self._SODWsp(X, target_neighbors)

        # Define optimization problem
        lmfun = lambda x: self._loss_grad(x)
        disp = 1 if self.verbose else None
        self.logger.info('Now optimizing...')
        self.n_funcalls = 0
        if self.save is not None:
            os.mkdir(self.temp_dir) if not os.path.exists(self.temp_dir) else None
            filename = self.save + '_' + str(self.n_funcalls) + '.npy'
            np.save(os.path.join(self.temp_dir, filename), self.L)

        L, loss, det = optimize.fmin_l_bfgs_b(func=lmfun, x0=self.L, bounds=None, m=100, pgtol=self.tol,
                                              maxfun=500*self.max_iter, maxiter=self.max_iter, disp=disp)
        self.details = det
        self.details['loss'] = loss
        self.L = L.reshape(L.size // D, D)

        return self

    def _loss_grad(self, L):
        """Compute the loss under a given L and the loss gradient w.r.t. L

        Args:
          L(array_like): [dxD,] the current linear transformation (flattened)

        Returns:
            tuple:

          float: the new loss
          array_like: [dxD,] the new gradient (flattened)

        """
        N, D = self.X.shape
        _, k = self.targets.shape
        self.L = L.reshape(L.size // D, D)
        self.n_funcalls += 1
        self.logger.info('Function call {}'.format(self.n_funcalls))
        if self.save is not None:
            filename = self.save + '_' + str(self.n_funcalls) + '.npy'
            np.save(os.path.join(self.temp_dir, filename), self.L)
        Lx = self.transform()

        # Compute distances to target neighbors under L (plus margin)
        self.logger.debug('Computing distances to target neighbors under new L...')
        dist_tn = np.zeros((N, k))
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
        A0 = sparse.csr_matrix((N, N))
        for nnid in reversed(range(k)):
            loss1 = np.maximum(dist_tn[imp1, nnid] - dist_imp, 0)
            act, = np.where(loss1 != 0)
            A1 = sparse.csr_matrix((2*loss1[act], (imp1[act], imp2[act])), (N, N))

            loss2 = np.maximum(dist_tn[imp2, nnid] - dist_imp, 0)
            act, = np.where(loss2 != 0)
            A2 = sparse.csr_matrix((2*loss2[act], (imp1[act], imp2[act])), (N, N))

            vals = np.squeeze(np.asarray(A2.sum(0) + A1.sum(1).T))
            A0 = A0 - A1 - A2 + sparse.csr_matrix((vals, (range(N), self.targets[:, nnid])), (N, N))
            loss = loss + np.sum(loss1 ** 2) + np.sum(loss2 ** 2)

        sum_outer_prods = self._SODWsp(self.X, A0, check=True)
        df = self.L @ (self.dfG + sum_outer_prods)
        df *= 2
        loss = loss + (self.dfG * (self.L.T @ self.L)).sum()
        self.logger.debug('Loss and gradient computed!\n')
        return loss, df.flatten()

    def _select_targets(self):
        """Compute target neighbors, that stay fixed during training

        Returns:
            array_like: [N,k] k neighbors for each input

        """
        k = self.k
        target_neighbors = np.empty((self.X.shape[0], k), dtype=int)
        for label in self.labels:
            ind, = np.where(np.equal(self.label_ind, label))
            dist = euclidean_distances(self.X[ind], squared=True)
            np.fill_diagonal(dist, np.inf)
            neigh_ind = np.argpartition(dist, k-1, axis=1)
            neigh_ind = neigh_ind[:, :k]
            # argpartition doesn't guarantee sorted order, so we sort again but only the k neighbors
            row_ind = np.arange(len(dist))[:, None]
            neigh_ind = neigh_ind[row_ind, np.argsort(dist[row_ind, neigh_ind])]
            target_neighbors[ind] = ind[neigh_ind]

        return target_neighbors

    def _find_impostors(self, Lx, margin_radii):
        """Compute all impostor pairs exactly

        Args:
          Lx (array_like):           [N,d] transformed inputs
          margin_radii (array_like): [N,] distances to the farthest target neighbors + margin

        Returns:
            tuple:

            (array_like, array_like, array_like): Impostor pairs and their distances

        """
        N = self.X.shape[0]

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

        # impostors = sparse.coo_matrix((np.ones(len(imp1)), (imp1, imp2)), (N, N), dtype=int)
        # imp1, imp2 = impostors.nonzero()
        idx = self._unique_pairs(imp1, imp2, N)

        # subsample constraints if they are too many
        if len(idx) > self.max_constr:
            idx = np.random.choice(idx, self.max_constr, replace=False)

        imp1 = np.asarray(imp1)[idx]
        imp2 = np.asarray(imp2)[idx]
        dist = np.asarray(dist)[idx]
        return imp1, imp2, dist

    def _find_impostors_sp(self, Lx, margin_radii):
        """Compute all impostor pairs exactly using a sparse matrix for storage

        Args:
          Lx (array_like):            [N, d] transformed inputs
          margin_radii (array_like):  [N,] distances to the farthest target neighbors + margin

        Returns:
            tuple:

            imp1 (array_like): [m,] sample indices
            imp2 (array_like): [m,] indices of impostors (samples that violate the margin) to imp1
            dist (array_like): [m,] pairwise distances of (imp1, imp2)
        """
        N = self.X.shape[0]

        # Initialize impostors matrix
        impostors_sp = sparse.csr_matrix((N, N), dtype=np.int8)
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
                new_imps = sparse.csr_matrix(([1]*len(imp1), (imp1, imp2)), shape=(N, N), dtype=np.int8)
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

        Args:
          x1 (array_like):      [n,] transformed inputs
          x2 (array_like):      [m,] transformed inputs, where always m < n
          t1 (array_like):      [n,] distances to margins
          t2 (array_like):      [m,] distances to margins
          batch_size (int):     size of each chunk of x1 to compute distances to (default: 500)
          return_dist (bool):  (default: False)

        Returns:
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
    def _SODWsp(x, weights, check=False):
        """Computes the sum of weighted outer products using a sparse weights matrix

        Args:
          x (array_like):       [N, D] N feature vectors of dimension D
          weights (csr_matrix): [N, N] target neighbors (adjacency-like) matrix
          check (bool):         whether to remove rows and columns of the symmetrized weights matrix that are zero (default: False)

        Returns:
            array_like:         [D, D] the sum of all weighted outer products

        """
        weights_sym = weights + weights.T
        if check:
            _, cols = weights_sym.nonzero()
            idx = np.unique(cols)
            weights_sym = weights_sym.tocsc()[:, idx].tocsr()[idx, :]
            x = x[idx]

        n = weights_sym.shape[0]
        diag = sparse.spdiags(weights_sym.sum(axis=0), 0, n, n)
        laplacian = diag.tocsr() - weights_sym
        sodw = x.T @ laplacian @ x
        return sodw

    @staticmethod
    def _cdist(X, a, b, batch_size=500):
        """Equivalent to  np.sum(np.square(x[a] - x[b]), axis=1)

        Args:
          X (array_like): [N, D] N feature vectors of dimension D
          a (array_like): [m,] indices of samples
          b (array_like): [m,] indices of other samples
          batch_size:  size of each chunk of X to compute distances for (default: 500)

        Returns:
            array-like: [m,] pairwise distances

        """
        n = len(a)
        res = np.zeros(n)
        for chunk in gen_batches(n, batch_size):
            res[chunk] = np.sum(np.square(X[a[chunk]] - X[b[chunk]]), axis=1)
        return res

    def _unique_pairs(self, i, j, n):
        """Find the unique pairs contained in zip(i, j)

        Args:
          i (list): m indices of samples
          j (list): m indices of impostors
          n (int): total number of samples (= maximum sample index + 1)

        Returns:
            array-like: k <= m indices of unique pairs

        """
        # First generate a hash array
        h = np.array([a * n + b for a, b in zip(i, j)])

        # Get the indices of the unique elements in the hash array
        _, idx = np.unique(h, return_index=True)
        self.logger.debug('Found {} unique pairs out of {}.'.format(len(idx), len(h)))
        return idx

    def _print_config(self):
        print('Parameters:\n')
        params_to_print = {'k', 'dim_out', 'max_iter', 'use_pca', 'max_constr', 'load', 'save', 'temp_dir', 'use_sparse', 'tol'}
        for k, v in self.__dict__.items():
            if k in params_to_print:
                print('{:10}: {}'.format(k, v))
        print()
