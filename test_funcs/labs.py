import torch
import numpy as np
from collections import OrderedDict
from .base import TestFunction


class LABS(TestFunction):
    def __init__(self, random_seed=0, normalize=True):
        super(LABS, self).__init__(normalize)
        self.seed = random_seed
        self.dim = 50
        self.n_vertices = np.array([2] * self.dim)
        self.config = self.n_vertices
        self.categorical_dims = np.arange(self.dim)
        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def compute(self, x, normalize=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.int()
        if x.dim() == 1:
            x = x.reshape(1, -1)
        res = torch.tensor([self._evaluate_single(x_, normalize) for x_ in x])
        # Add a small ammount of noise to prevent training instabilities
        # res += 1e-6 * torch.randn_like(res)
        return res

    def _evaluate_single(self, x, normalize=None):
        assert x.dim() == 1
        if x.dim() == 2:
            x = x.squeeze(0)
        assert x.shape[0] == self.dim
        x = x.cpu().numpy()
        N = x.shape[0]  # x[x == 0] = -1
        E = 0.0  # energy
        for k in range(1, N):
            C_k = 0
            for j in range(0, N - k - 1):
                C_k += (-1) ** (1 - x[j] * x[j + k])
            E += C_k**2
        if E == 0:
            print("found zero")
        res = N / (2 * E)
        if normalize:
            assert self.mean is not None and self.std is not None
            res = (res - self.mean) / self.std
        return -1 * res  # Flip the sign since Casmo minimizes
