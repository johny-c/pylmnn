from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
import numpy as np


class Store(ABC):

    @abstractmethod
    def extend(self, row, col, dist=None):
        pass

    @abstractmethod
    def get(self):
        pass


class ListStore(Store):

    def __init__(self):
        self.row = []
        self.col = []
        self.dist = []

    def extend(self, row, col, dist=None):
        self.row.extend(row)
        self.col.extend(col)
        self.dist.extend(dist)

    def get(self):
        imp_row = np.asarray(self.row, dtype=np.intp)
        imp_col = np.asarray(self.col, dtype=np.intp)
        imp_dist = np.asarray(self.dist)
        return imp_row, imp_col, imp_dist


class SparseMatrixStore(Store):

    def __init__(self, n_samples, dtype=np.int8):
        self.n_samples = n_samples
        self.dtype = dtype
        self._data = csr_matrix((n_samples, n_samples), dtype=dtype)

    def extend(self, row, col, dist=None):
        n = self.n_samples
        v = np.ones(len(row), dtype=self.dtype)
        new_data = csr_matrix((v, (row, col)), dtype=self.dtype, shape=(n, n))

        self._data = self._data + new_data

    def get(self):
        graph = self._data.tocoo(copy=False)
        return graph.row, graph.col, None
