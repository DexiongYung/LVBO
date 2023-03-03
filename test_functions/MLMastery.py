import torch
from math import pi

import torch
from numpy.random import normal


class MLMastery:
    def __init__(self):
        self.optimal_X = (0.9)
        self.optimal_y = (0.810)
        self.X_range = torch.Tensor([[0], [1]])

    def forward(self, x, noise: float = 0.1):
        noise = normal(loc=0, scale=noise)
        return (x ** 2 * torch.sin(5 * pi * x) ** 6.0) + noise
