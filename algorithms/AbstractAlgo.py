import torch
from torch import Tensor
from abc import abstractmethod
from botorch.fit import fit_gpytorch_mll


class AbstractAlgo:
    def __init__(self, X_cont: Tensor, X_qual: Tensor, Y: Tensor, qual_levels: list, X_cont_bounds: Tensor,
                 device: str):
        self.X_cont = X_cont
        self.X_qual = X_qual
        self.Y = Y.to(device)
        self.num_quant = X_cont.shape[1]
        self.num_qual = X_qual.shape[1]
        self.qual_levels = qual_levels
        self.X_cont_bounds = X_cont_bounds
        self.device = device

        self.mll = None
        self.acq = None
        self.model = None
        self.kernel = None
        self.X_latent_bounds = None

    def set_marginal_ll(self):
        raise NotImplemented("Not implemented, required in order to run BO")

    def get_marginal_ll(self):
        if self.mll:
            return self.mll
        else:
            self.set_marginal_ll()

    @abstractmethod
    def set_model(self):
        raise NotImplemented("Not implemented, required in order to run BO")

    def get_model(self):
        if self.model:
            return self.model
        else:
            self.set_model()

    @abstractmethod
    def set_acquisition(self):
        raise NotImplemented("Not implemented, required in order to run BO")

    @abstractmethod
    def convert_qual_to_quant(self, X: Tensor):
        raise NotImplemented()

    def get_acquisition(self):
        if self.acq:
            return self.acq
        else:
            self.set_acquisition()

    def fit_mll(self):
        fit_gpytorch_mll(self.get_marginal_ll())

    def optimize_acq(self, X: Tensor, num_iter: int = 50, lr: float = 0.1):
        X.requires_grad_(True)
        optimizer = torch.optim.Adam([X], lr=lr)
        X_traj = []
        full_bounds = torch.cat((self.X_cont_bounds, self.X_latent_bounds), dim=1)

        for i in range(num_iter):
            optimizer.zero_grad()
            losses = - self.acq(X)
            loss = losses.sum()

            loss.backward()
            optimizer.step()

            for i in range(full_bounds.shape[1]):
                self.X_full.data[:, i].clamp_(full_bounds[0, i], full_bounds(1, i))

            X_traj.append(self.X_full.detach().clone())

        return X_traj

    def add_to_DOE(self, candidates: Tensor, Y: Tensor):
        X_cont_cand = candidates[:, :self.num_quant]
        X_qual_cand = candidates[:, self.num_quant, :]
        self.X_cont = self.X_cont.cat(X_cont_cand)
        self.X_qual = self.X_qual.cat(X_qual_cand)
        self.Y = self.Y.cat(Y)
        best_idx = Y.argmax()
        self.best_observations.append(dict(cont=X_cont_cand[best_idx], qual=X_qual_cand[best_idx], Y=Y[best_idx]))

    def get_DOE_as_pd(self):
        # TODO: Might need to convert back to Pd
        pass
