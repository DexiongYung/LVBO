import torch
from kernels.LVKernel import LVKernel, LatentVariableParameters
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

    def fit_mll(self, epoch: int = 50, lr: float = 0.1):
        optimizer = torch.optim.SGD(list(self.model.parameters()) + self.latent_params_class.latent_params, lr=lr)

        self.model.train()

        X = torch.column_stack((self.X_cont, self.latent_params_class.convert_qual_to_latent(X_qual=self.X_qual)))

        for i in range(epoch):
            # clear gradients
            optimizer.zero_grad()
            # forward pass through the model to obtain the output MultivariateNormal
            output = self.model(X)
            # Compute negative marginal log likelihood
            loss = - self.mll(output, self.model.train_targets)
            # back prop gradients
            loss.backward()
            # print every 10 iterations
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch {i + 1:>3}/{epoch} - Loss: {loss.item():>4.3f} "
                    f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():>4.3f} "
                    f"noise: {self.model.likelihood.noise.item():>4.3f}"
                )
            optimizer.step()

    def set_marginal_ll(self):
        self.mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)

    def set_model(self):
        self.latent_params_class = LatentVariableParameters(num_qual=self.X_qual.shape[1], qual_levels=self.qual_levels)
        self.kernel = LVKernel(num_cont=self.X_cont.shape[1],
                               latent_feat_nums=self.latent_params_class.latent_feat_nums)
        self.X_latent_bounds = self.latent_params_class.X_latent_bounds
        self.X_latent = self.latent_params_class.convert_qual_to_latent(X_qual=self.X_qual)
        self.model = SingleTaskGP(train_X=torch.column_stack((self.X_cont, self.X_latent)), train_Y=self.Y,
                                  covar_module=self.kernel)
