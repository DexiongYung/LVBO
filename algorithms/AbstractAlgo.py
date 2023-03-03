import torch
from torch import Tensor
from abc import abstractmethod
from botorch.optim import optimize_acqf
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

    def optimize_acq(self):
        raise NotImplemented()

    def add_to_DOE(self, candidates: Tensor, Y: Tensor):
        X_cont_cand = candidates[:, :self.num_cont]
        X_qual_cand = candidates[:, self.num_cont:]
        self.X_cont = torch.cat((self.X_cont, X_cont_cand))
        self.X_qual = torch.cat((self.X_qual, X_qual_cand))
        self.Y = torch.cat((self.Y, Y.unsqueeze(1)))

    def get_DOE_as_pd(self):
        # TODO: Might need to convert back to Pd
        pass
