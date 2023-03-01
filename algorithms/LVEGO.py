import torch
from botorch import fit_gpytorch_mll
from kernels.LVKernel import LVKernel
from algorithms.AbstractAlgo import AbstractAlgo
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.sampling.stochastic_samplers import StochasticSampler


class LV_EGO(AbstractAlgo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_model()
        self.set_marginal_ll()
        self.set_acquisition()

    # TODO: Probably need to create custom act opt function version of botorch.optim.optimize.optimize_acqf_list
    # Source: https://botorch.org/api/optim.html
    def set_acquisition(self):
        self.acq = ExpectedImprovement(model=self.model, best_f=self.Y.min(),
                                       sampler=StochasticSampler(sample_shape=torch.Size([128])))

    def fit_mll(self):
        self.mll.to(torch.column_stack((self.X_cont, self.X_qual)))
        fit_gpytorch_mll(self.mll)

    def set_marginal_ll(self):
        self.mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)

    def set_model(self):
        self.kernel = LVKernel(num_cont=self.X_cont.shape[1], num_qual=self.num_qual, qual_levels=self.qual_levels)
        self.X_latent_bounds = self.kernel.X_latent_bounds
        self.model = SingleTaskGP(train_X=torch.column_stack((self.X_cont, self.X_qual)), train_Y=self.Y,
                                  covar_module=self.kernel)
