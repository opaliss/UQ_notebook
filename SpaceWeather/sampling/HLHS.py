""" Code to run Hierarchical Latin Hypercube sampling
https://stackoverflow.com/questions/72638189/how-can-i-continue-latin-hypercube-sampling-in-python

Last Modified: Oct 23rd, 2022 [OI]
"""
import numpy as np
from skopt.sampler import Lhs
from sklearn.utils import check_random_state
from scipy import spatial
from skopt.space import Space


def _random_permute_matrix(h, random_state=None):
    rng = check_random_state(random_state)
    h_rand_perm = np.zeros_like(h)
    samples, n = h.shape
    for j in range(n):
        order = rng.permutation(range(samples))
        h_rand_perm[:, j] = h[order, j]
    return h_rand_perm


class LHS_extendable(Lhs):
    def __init__(self, lhs_type="centered", criterion="maximin", iterations=1000):
        self.lhs_type = lhs_type
        self.criterion = criterion
        self.iterations = iterations

    def generate(self, dimensions, n_samples, existing_samples=None, random_state=None):
        rng = check_random_state(random_state)
        space = Space(dimensions)
        transformer = space.get_transformer()
        n_dim = space.n_dims
        space.set_transformer("normalize")
        if existing_samples is not None:
            existing_samples = space.transform(existing_samples)
        if self.criterion is None or n_samples == 1:
            h = self._lhs_normalized(n_dim, n_samples, existing_samples, rng)
            h = space.inverse_transform(h)
            space.set_transformer(transformer)
            return h
        else:
            h_opt = self._lhs_normalized(n_dim, n_samples, existing_samples, rng)
            h_opt = space.inverse_transform(h_opt)
            if self.criterion == "correlation":
                mincorr = np.inf
                for i in range(self.iterations):
                    # Generate a random LHS
                    h = self._lhs_normalized(n_dim, n_samples, existing_samples, rng)
                    r = np.corrcoef(np.array(h).T)
                    if len(np.abs(r[r != 1])) > 0 and \
                            np.max(np.abs(r[r != 1])) < mincorr:
                        mincorr = np.max(np.abs(r - np.eye(r.shape[0])))
                        h_opt = h.copy()
                        h_opt = space.inverse_transform(h_opt)
            elif self.criterion == "maximin":
                maxdist = 0
                # Maximize the minimum distance between points
                for i in range(self.iterations):
                    h = self._lhs_normalized(n_dim, n_samples, existing_samples, rng)
                    d = spatial.distance.pdist(np.array(h), 'euclidean')
                    if maxdist < np.min(d):
                        maxdist = np.min(d)
                        h_opt = h.copy()
                        h_opt = space.inverse_transform(h_opt)
            elif self.criterion == "ratio":
                minratio = np.inf

                # Maximize the minimum distance between points
                for i in range(self.iterations):
                    h = self._lhs_normalized(n_dim, n_samples, existing_samples, rng)
                    p = spatial.distance.pdist(np.array(h), 'euclidean')
                    if np.min(p) == 0:
                        ratio = np.max(p) / 1e-8
                    else:
                        ratio = np.max(p) / np.min(p)
                    if minratio > ratio:
                        minratio = ratio
                        h_opt = h.copy()
                        h_opt = space.inverse_transform(h_opt)
            else:
                raise ValueError("Wrong criterion."
                                 "Got {}".format(self.criterion))
            space.set_transformer(transformer)
            return h_opt

    def _lhs_normalized(self, n_dim, n_samples, existing_samples, random_state):
        rng = check_random_state(random_state)
        x = np.linspace(0, 1, n_samples + 1)
        if existing_samples is None:
            u = rng.rand(n_samples, n_dim)
            h = np.zeros_like(u)
        else:
            u = rng.rand(n_samples - len(existing_samples), n_dim)
            h = np.zeros_like(u)

        if self.lhs_type == "centered":
            for j in range(n_dim):
                x_dim = x
                if existing_samples is not None:
                    existing_samples = np.array(existing_samples)
                    total_removed = 0
                    for old_sample in existing_samples:
                        # Find quadrant where old sample is located and remove that position
                        for i, position in enumerate(x_dim):
                            if 0 <= old_sample[j] - position < np.diff(x_dim)[i]:
                                x_dim = np.delete(x_dim, i, 0)

                h[:, j] = np.diff(x)[0] / 2.0 + x_dim[:-1]
        elif self.lhs_type == "classic":
            for j in range(n_dim):
                x_dim = x
                if existing_samples is not None:
                    existing_samples = np.array(existing_samples)
                    total_removed = 0
                    for old_sample in existing_samples:
                        # Find quadrant where old sample is located and remove that position
                        for i, position in enumerate(x_dim):
                            if 0 <= old_sample[j] - position < np.diff(x_dim)[i]:
                                x_dim = np.delete(x_dim, i, 0)
                h[:, j] = u[:, j] * np.diff(x)[0] + x_dim[:-1]
        else:
            raise ValueError("Wrong lhs_type. Got ".format(self.lhs_type))

        # Remove new samples in the same quadrant as old samples
        random_matrix = _random_permute_matrix(h, random_state=rng)

        if existing_samples is not None:
            random_matrix = np.concatenate((random_matrix, existing_samples), axis=0)
        return random_matrix
