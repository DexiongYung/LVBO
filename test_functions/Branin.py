import torch
import math


class Branin:
    def __init__(self):
        self.qual_levels = [[0, 0.333, 0.666, 1]]
        self.continuous_vars = ['X_cont']
        self.qualitative_vars = ['X_qual']
        self.continuous_range = [(0, 1)]
        self.optimal = dict(X_cont=0.182, X_qual=0.666, Y=2.791)
        self.x_min = dict(X_cont=-5, X_qual=0)
        self.x_max = dict(X_cont=10, X_qual=15)
        self.X_cont_bounds_tnsr = torch.Tensor([[0], [1]])

    def forward(self, X_cont: torch.Tensor, X_qual: torch.Tensor):
        # if not is_tensor_in_qual_levels(X_qual=X_qual, qual_levels=self.qual_levels):
        #     raise ValueError(fr"X_qual={X_qual}. X_qual has value \not\in {self.qual_levels}")
        #
        # if not is_tensor_in_cont_bounds(X_cont=X_cont, cont_bounds=self.continuous_range):
        #     raise ValueError(fr"X_cont={X_cont}. X_cont \not\in [-5,10]")

        b = (5 / (4 * math.pi ** 2))
        c = 5 / math.pi
        r = 6
        s = 10
        t = 1 / (8 * math.pi)
        X_cont = self.x_min['X_cont'] + (self.x_max['X_cont'] - self.x_min['X_cont']) * X_cont
        X_qual = self.x_min['X_qual'] + (self.x_max['X_qual'] - self.x_min['X_qual']) * X_qual

        return -1 * ((X_qual - b * torch.pow(X_cont, 2) + c * X_cont - r) ** 2 + s * (1 - t) * torch.cos(X_cont) + s)
