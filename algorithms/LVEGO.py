import torch
from kernels.LVKernel import LVKernel
from algorithms.AbstractAlgo import AbstractAlgo
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
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
        self.acq = qExpectedImprovement(model=self.model, best_f=self.Y.min(),
                                        sampler=StochasticSampler(sample_shape=torch.Size([128])))

    def fit_mll(self):
        NUM_EPOCHS = 150

        self.model.train()


        for epoch in range(NUM_EPOCHS):
            # clear gradients
            self.kernel.optimizer.zero_grad()
            # forward pass through the model to obtain the output MultivariateNormal
            output = self.model(torch.column_stack((self.X_cont, self.kernel.convert_qual_to_latent(self.X_qual))))
            # Compute negative marginal log likelihood
            loss = - self.mll(output, self.model.train_targets)
            # back prop gradients
            loss.backward()
            # print every 10 iterations
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
                )
            self.kernel.optimizer.step()

    def set_marginal_ll(self):
        self.mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)

    def set_model(self):
        self.kernel = LVKernel(num_cont=self.X_cont.shape[1], num_qual=self.num_qual, qual_levels=self.qual_levels)
        self.X_qual_bounds = self.kernel.X_qual_bounds
        self.X_latent = self.kernel.convert_qual_to_latent(self.X_qual)
        self.model = SingleTaskGP(train_X=torch.column_stack((self.X_cont, self.X_latent)), train_Y=self.Y,
                                  covar_module=self.kernel)
