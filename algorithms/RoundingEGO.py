import torch
from torch import Tensor
from algorithms.AbstractAlgo import AbstractAlgo
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.sampling.stochastic_samplers import StochasticSampler


class Rounding_EGO(AbstractAlgo):
    def __init__(self, X_qual_bounds: Tensor, **kwargs):
        super().__init__(self, **kwargs)
        self.set_model()
        self.set_marginal_ll()
        self.set_acquisition()
        self.X_qual_bounds = X_qual_bounds

    def set_acquisition(self):
        self.acq = ExpectedImprovement(model=self.model, best_f=self.Y.min(),
                                       sampler=StochasticSampler(sample_shape=torch.Size([128])))

    def set_marginal_ll(self):
        self.mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)

    def set_model(self):
        self.X_qual_bounds = self.kernel.X_latent_bounds
        self.model = SingleTaskGP(train_X=torch.column_stack((self.X_cont, self.X_qual)), train_Y=self.Y)
