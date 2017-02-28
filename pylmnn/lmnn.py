import numpy as np
from numpy import linalg as LA
from scipy import sparse, optimize
from sklearn.metrics import pairwise as pw
import logging, sys, os


class LMNN:
    """
    Large Margin Nearest Neighbor metric learning.

    This implementation follows closely Kilian Weinberger's MATLAB code found at
    https://bitbucket.org/mlcircus/lmnn
    which solves the unconstrained problem, finding a linear transformation with L-BFGS instead of
    solving the constrained problem that finds the globally optimal metric.

    Copyright (c) 2017, John Chiotellis
    Licensed under the GPLv3 license (see LICENSE.txt)
    """
    def __init__(self, L=None, k=3, max_iter=200, use_pca=True, tol=1e-5, verbose=True, outdim=None, save=None, loglevel=logging.INFO):
        """
        Instantiate the LMNN classifier
        :param L:           dxD matrix, initial transformation, if None identity or pca will be used
                            based on use_pca (default: None)
        :param k:           scalar, number of target neighbors (default: 3)
        :param max_iter:    scalar, maximum number of iterations in the optimization (default: 200)
        :param use_pca:     flag, if True use pca to initialize the transformation,
                            otherwise identity is used except if an L is given  (default: True)
        :param tol:         scalar, tolerance for the optimization  (default: 1e-5)
        :param verbose:     flag, output information during the optimization (default:True)
        :param outdim:      flag, preferred dimensionality of the inputs after the
                            transformation, if None it is inferred from use_pca and L (default:None)
        :param save:        string, if not None save the intermediate linear transformations in a
                            folder with this string as filename (default: None)
        :param loglevel:    logging level of verbosity for debugging purposes (default: INFO)
        """
        self.params = dict(k=k, max_iter=max_iter, use_pca=use_pca, tol=tol, verbose=verbose, outdim=outdim)
        self.L = L
        self.targets = None
        self.dfG = None
        self.save = save
        self.iter = 0
        self.loglevel = loglevel
        self.tempdir = 'temp_res'
        logging.basicConfig(stream=sys.stdout, level=self.loglevel)

    def transform(self, X=None):
        if X is None:
            X = self.X
        return X @ self.L.T

    def _check_inputs(self, X, labels):
        assert len(labels) == X.shape[0], "Number of labels ({}) does not match the number of " \
                                          "points ({})!".format(len(labels), X.shape[0])
        unique_labels, self.label_idx = np.unique(labels, return_inverse=True)
        self.labels = np.arange(len(unique_labels))
        max_k = np.bincount(self.label_idx).min() - 1

        if self.params['k'] > max_k:
            print('Warning: K too high. Setting K={}\n'.format(max_k))
            self.params['k'] = max_k

        self.X = X

    def _init_transformer(self):
        if self.L is not None: return
        if self.params['use_pca']:
            cc = np.cov(self.X, rowvar=False)  # Mean is removed
            evals, evecs = LA.eigh(cc)  # Get evals in ascending order, evecs in columns
            evecs = np.fliplr(evecs)    # Flip evecs to get them in descending eigenvalue order
            self.L = evecs.T            # Set L rows equal to eigenvectors
        else:
            self.L = np.eye(self.X.shape[1])

        outdim = self.params['outdim']
        if outdim is not None:
            D = self.X.shape[1]
            if outdim > self.L.shape[0]:
                print('outdim({}) cannot be larger than the inputs dimensionality ({}), '
                      'setting outdim to {}!'.format(outdim, D, D))
                outdim = D
            self.L = self.L[:outdim]

    def fit(self, X, labels):
        verbose = self.params['verbose']
        tol = self.params['tol']
        max_iter = self.params['max_iter']

        # Check data consistency and initialize label counts
        self._check_inputs(X, labels)
        k = self.params['k']
        print('Parameters:\n')
        [print('{:10}: {}'.format(k, v)) for k, v in self.params.items()]

        # Initialize L
        self._init_transformer()

        # Find target neighbors (fixed)
        logging.info('Finding target neighbors...')
        self.targets = self._select_targets()

        # Compute gradient component of target neighbors (constant)
        logging.info('Computing gradient component due to target neighbors...')
        N, D = X.shape
        rows = np.repeat(np.arange(N), k)  # 0 0 0 1 1 1 2 2 2 ... (n-1) (n-1) (n-1) with k=3
        cols = self.targets.flatten()
        target_neighbors = sparse.csr_matrix((np.ones(N*k), (rows, cols)), shape=(N, N))
        self.dfG = self._SODWsp(X, target_neighbors)

        # Define optimization problem
        lmfun = lambda x: self._loss_grad(x)
        disp = 1 if verbose else None
        logging.info('Now optimizing...')
        self.iter = 0
        if self.save is not None:
            os.mkdir(self.tempdir) if not os.path.exists(self.tempdir) else None
            filename = self.save + '_' + str(self.iter) + '.npy'
            np.save(os.path.join(self.tempdir, filename), self.L)

        L, loss, det = optimize.fmin_l_bfgs_b(func=lmfun, x0=self.L, bounds=None, m=100, pgtol=tol,
                                              maxfun=500*max_iter, maxiter=max_iter, disp=disp)

        print('Finished!')
        self.L = L.reshape(L.size // D, D)
        return self, loss, det

    def _loss_grad(self, L):
        """
        Compute the loss under a given L and the loss gradient w.r.t. L
        :param L:   dxD flat, the current linear transformation
        :return:    scalar, new loss and dxD flat, new gradient
        """
        N, D = self.X.shape
        _, k = self.targets.shape
        self.L = L.reshape(L.size // D, D)
        self.iter += 1
        logging.info('Iteration {}'.format(self.iter))
        if self.save is not None:
            filename = self.save + '_' + str(self.iter) + '.npy'
            np.save(os.path.join(self.tempdir, filename), self.L)
        Lx = self.transform()

        # Compute distances to target neighbors under L (plus margin)
        logging.debug('Computing distances to target neighbors under new L...')
        dist_tn = np.zeros((N, k))
        for j in range(k):
            dist_tn[:, j] = np.sum(np.square(Lx - Lx[self.targets[:, j]]), axis=1) + 1

        # Compute distances to impostors under L
        logging.debug('Setting margin radii...')
        margin_radii = np.add(dist_tn[:, -1], 2)
        imp1, imp2, dist_imp = self._find_impostors(Lx, margin_radii)
        # logging.debug('Computing distances to impostors under new L...')
        # dist_imp = self._cdist(Lx, imp1, imp2)

        logging.debug('Computing loss and gradient under new L...')
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
        logging.debug('Loss and gradient computed!\n')
        return loss, df.flatten()

    def _select_targets(self):
        """
        Compute target neighbors, that stay fixed throughout the algorithm
        :return:    Nxk matrix with k neighbors for each input
        """
        k = self.params['k']
        target_neighbors = np.empty((self.X.shape[0], k), dtype=int)
        for label in self.labels:
            ind, = np.where(np.equal(self.label_idx, label))
            dist = pw.euclidean_distances(self.X[ind], squared=True)
            np.fill_diagonal(dist, np.inf)
            neigh_ind = np.argpartition(dist, k-1, axis=1)
            neigh_ind = neigh_ind[:, :k]
            # argpartition doesn't guarantee sorted order, so we sort again but only the k neighbors
            row_ind = np.arange(len(dist))[:, None]
            neigh_ind = neigh_ind[row_ind, np.argsort(dist[row_ind, neigh_ind])]
            target_neighbors[ind] = ind[neigh_ind]

        return target_neighbors

    def _find_impostors(self, Lx, margin_radii):
        """
        Compute all impostor pairs exactly
        :param Lx:              Nxd transformed inputs matrix
        :param margin_radii:    Nx1 vector of distances to the farthest target neighbors + margin
        :return: Px1 vectors imp1 and imp2, samples that violate the margin of other sample(s)
        """
        N = self.X.shape[0]

        # Initialize impostors vectors
        imp1, imp2, dist = [], [], []
        logging.debug('Now computing impostor vectors...')
        for label in self.labels[:-1]:
            idx_in, = np.where(np.equal(self.label_idx, label))
            idx_out, = np.where(np.greater(self.label_idx, label))
            # Permute the indices (experimental)
            # idx_in = np.random.permutation(idx_in)
            # idx_out = np.random.permutation(idx_out)

            # Subdivide idx_out x idx_in to chunks of a size that is fitting in memory (1 MB)
            logging.debug('Impostor classes {} to class {}..'.
                          format(self.labels[self.labels > label], label))
            ii, jj, dd = self._find_imps(Lx[idx_out], Lx[idx_in],
                                         margin_radii[idx_out], margin_radii[idx_in])
            if len(ii):
                imp1.extend(idx_out[ii])
                imp2.extend(idx_in[jj])
                dist.extend(dd)

        # impostors = sparse.coo_matrix((np.ones(len(imp1)), (imp1, imp2)), (N, N), dtype=int)
        # imp1, imp2 = impostors.nonzero()
        idx = self._unique_pairs(imp1, imp2, N)
        imp1 = np.asarray(imp1)[idx]
        imp2 = np.asarray(imp2)[idx]
        dist = np.asarray(dist)[idx]
        return imp1, imp2, dist

    @staticmethod
    def _find_imps(x1, x2, t1, t2, mem_budget=1e7):
        """
        Find impostor pairs in a minibatch fashion to avoid large memory usage
        :param x1: nx1 vector of transformed inputs
        :param x2: mx1 vector of transformed inputs, where always m < n
        :param t1: nx1 vector of distances to margins
        :param t2: mx1 vector of distances to margins
        :param mem_budget: memory budget (in bytes) for intermediate distance computations (
        default: 1e7 = 10 MB)
        :return: px1, px1, impostor pairs vectors
        """
        n, m = len(t1), len(t2)
        minibatch_size = int(mem_budget / (8 * m))
        im1, im2, dists = [], [], []
        for i in range(0, n, minibatch_size):
            bb = min(minibatch_size, n - i)
            dist_out_in = pw.euclidean_distances(x1[i:i + bb], x2, squared=True)
            i1, j1 = np.where(dist_out_in < t1[i:i + bb, None])
            i2, j2 = np.where(dist_out_in < t2[None, :])
            if len(i1):
                im1.extend(i1 + i)
                im2.extend(j1)
                dists.extend(dist_out_in[i1, j1])
            if len(i2):
                im1.extend(i2 + i)
                im2.extend(j2)
                dists.extend(dist_out_in[i2, j2])

        return im1, im2, dists

    @staticmethod
    def _SODWsp(x, weights, check=False):
        """
        Computes the sum of weighted outer products using a sparse weights matrix
        :param x:           NxD matrix consisting of N row vectors
        :param weights:     NxN csr_matrix, target neighbors
        :param check:       flag, if True rows and columns of the symmetrized weights matrix that
                            are zero are removed (default: False)
        :return:            DxD the sum of all weighted outer products
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
    def _cdist(x, a, b, mem_budget=1e7):
        """ Equivalent to  np.sum(np.square(x[a] - x[b]), axis=1) """
        n = len(a)
        d = x.shape[1]
        minibatch_size = int(mem_budget / (8 * d))
        res = np.zeros(n)
        for i in range(0, n, minibatch_size):
            bb = min(minibatch_size, n - i)
            # ind = k:min(k + minibatch_size - 1, n)
            res[i:i+bb] = np.sum(np.square(x[a[i:i+bb]] - x[b[i:i+bb]]), axis=1)
        return res

    @staticmethod
    def _unique_pairs(i, j, n):
        # First generate a hash array
        h = np.array([a * n + b for a, b in zip(i, j)])

        # Get the indices of the unique elements in the hash array
        _, idx = np.unique(h, return_index=True)
        logging.debug('Found {} unique pairs out of {}.'.format(len(idx), len(h)))
        return idx

    def load_stored(self, iteration):
        """
        Loads a linear transformation from the temporary results directory
        :param iteration: Load the saved L from this iteration
        :return:    the saved L
        """
        filename = self.save + '_' + str(iteration) + '.npy'
        self.L = np.load(os.path.join(self.tempdir, filename))
        return self.L
