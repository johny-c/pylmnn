"""Tools to assist in sampling impostors for PyLMNN. """
from abc import ABC, abstractmethod
import numpy as np


class Sampler(ABC):
    @abstractmethod
    def extend(self, latest_values):
        pass

    @abstractmethod
    def sample(self):
        pass


class UniformSampler(Sampler):
    """Post-hoc uniform sample of fixed size from a stream values.

    Parameters
    ----------
    sample_size : int
        Number of elements to take as the sample.

    random_state : numpy.RandomState
        A pseudo random number generator object used for sampling.
    """

    def __init__(self, sample_size, random_state):
        self.sample_size = sample_size
        self.random_state = random_state

        self._values = []

    def extend(self, latest_values):
        """Adds elements to be sampled from.

        Parameters
        ----------
        latest_values : array-like, shape (n_samples,)
            The latest part of the stream to sample from.

        """
        self._values.extend(latest_values)

    def sample(self):
        """Gets the current sample.

        Returns
        -------
        sample : array, shape (sample_size,)
            The current sample from the stream

        """
        n = len(self._values)
        k = self.sample_size
        if n > k:
            ind = self.random_state.choice(n, k, replace=False)
            return np.asarray(self._values, dtype=object)[ind]
        else:
            return self._values


class ReservoirSampler(Sampler):
    """Pseudo-uniform online sample of fixed size from a stream of values.

    Parameters
    ----------
    sample_size : int
        Number of elements to take as the sample.

    random_state : numpy.RandomState
        A pseudo random number generator object used for sampling.

    References
    ----------

    .. [1] Li, Kim-Hung (4 December 1994).
           "Reservoir-Sampling Algorithms of Time Complexity O(n(1+log(N/n)))"
           ACM Transactions on Mathematical Software. 20 (4): 481–493.
    """

    def __init__(self, sample_size, random_state):
        self.sample_size = sample_size
        self.random_state = random_state

        self._i = 0
        self._reservoir = []

        u = random_state.random()
        self._w = np.exp(np.log(u) / sample_size)

        u = random_state.random()
        self._next_i = int(sample_size + np.log(u) / np.log(1 - self._w) + 1)

        # stores randomly generated values for sampling which we do in batches
        # to minimize overhead of getting random numbers
        # precomputing and  storing these per-sampled-element values is over
        # twice as fast as naively generating the random numbers and computing
        # them inside the sampling loop
        self._rand_replacement = None
        self._rand_w_multiplier = None
        self._rand_next_i_partial = None
        self._rand_ind = sample_size

    def extend(self, latest_values):
        """Adds elements to the reservoir sample.

        Parameters
        ----------
        latest_values : array-like, shape (n_samples,)
            The latest part of the stream to sample from.

        """

        offset = self._i
        k = self.sample_size
        rng = self.random_state

        latest_values = list(latest_values)  # TODO: This seems unnecessary
        remaining_capacity = k - offset

        # if reservoir not full, copy from latest_values to fill
        if remaining_capacity > 0:
            to_copy = min(remaining_capacity, len(latest_values))
            self._reservoir.extend(latest_values[:to_copy])

        # NOTE: At this point, the reservoir might still be not full

        # general case: sample from latest_values
        while self._next_i - offset < len(latest_values):
            if self._rand_ind >= k:
                # refill random numbers in batches to minimize overhead
                # also pre-compute some of the arithmetic to minimize overhead
                # doing this takes ReservoirSampler from taking about 5x as
                # long as numpy.RandomState.choice to ~2.5x as long
                self._rand_replacement = rng.randint(0, k, k)
                self._rand_w_multiplier = np.exp(np.log(rng.random(k)) / k)
                self._rand_next_i_partial = np.log(rng.random(k))
                self._rand_ind = 0

            # replace a random item of the reservoir with item i
            ind_src = self._next_i - offset
            ind_dst = self._rand_replacement[self._rand_ind]
            self._reservoir[ind_dst] = latest_values[ind_src]

            # book-keeping
            self._w *= self._rand_w_multiplier[self._rand_ind]
            s = self._rand_next_i_partial[self._rand_ind]
            self._next_i = int(self._next_i + 1 + s / np.log(1 - self._w))
            self._rand_ind += 1

        self._i += len(latest_values)

    def sample(self):
        """Gets the current reservoir sample.

        Returns
        -------
        sample : array, shape (sample_size,)
            The current sample from the stream

        """

        return self._reservoir
