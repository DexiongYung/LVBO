import torch
from torch import Tensor
from algorithms.AbstractAlgo import AbstractAlgo
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.acquisition.monte_carlo import qExpectedImprovement


class Rounding_EGO(AbstractAlgo):
    def __init__(self, X_cont, X_qual, Y, X_cont_bounds, X_qual_bounds: Tensor, X_qual_levels: list, **kwargs):
        self.X_qual_bounds = X_qual_bounds
        self.X_cont = X_cont
        self.X_qual = X_qual
        self.X_cont_bounds = X_cont_bounds
        self.Y = Y
        self.set_model()
        self.set_marginal_ll()
        self.set_acquisition()
        self.num_cont = X_cont.shape[1]
        self.num_qual = X_qual.shape[1]
        self.X_qual_levels = X_qual_levels
        self.level_spacing = (X_qual_levels[0][1] - X_qual_levels[0][0]) / 2

    def set_acquisition(self):
        self.acq = ExpectedImprovement(model=self.model, best_f=self.Y.max(),
                                       sampler=StochasticSampler(sample_shape=torch.Size([128])))

    def set_marginal_ll(self):
        self.mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)

    def fit_mll(self):
        fit_gpytorch_mll(self.mll)

    def optimize_acq(self):
        acq_func = qExpectedImprovement(model=self.model, best_f=standardize(self.Y).max())
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(self.num_qual + self.num_cont),
                torch.ones(self.num_qual + self.num_cont),
            ]),
            q=4,
            num_restarts=2,
            raw_samples=3,
        )

        # observe new values
        x_qual_new = unnormalize(candidates[:, self.num_cont:].detach(), bounds=self.X_qual_bounds)
        x_qual_new_copy = torch.clone(x_qual_new)
        for i in range(self.num_qual):
            curr_levels = self.X_qual_levels[i]
            for level in curr_levels:
                mask = self.level_spacing >= torch.abs(x_qual_new_copy[:, i] - level)
                x_qual_new[mask] = level

        return torch.column_stack((candidates[:, self.num_cont:], x_qual_new))

    def set_model(self):
        self.model = SingleTaskGP(train_X=normalize(torch.column_stack((self.X_cont, self.X_qual)),
                                                    torch.column_stack((self.X_cont_bounds, self.X_qual_bounds))),
                                  train_Y=standardize(self.Y))

    def run_iteration_of_BO(self):
        self.set_model()
        self.set_marginal_ll()
        self.fit_mll()
        return self.optimize_acq()
