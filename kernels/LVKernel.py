import torch
import gpytorch
from torch import Tensor


class LatentVariableParameters(object):
    def __init__(self, num_qual: int, qual_levels: list):
        self.num_latent = 0
        self.latent_params = list()
        self.latent_feat_nums = list()
        self.num_qual = num_qual
        self.qual_levels = qual_levels

        for i in range(num_qual):
            # TODO: Don't optimize level 1 and level 2 index 0
            latent_dim = 2 if len(qual_levels[i]) > 2 else 1
            self.num_latent += latent_dim
            self.latent_feat_nums.append(latent_dim)
            self.latent_params.append(torch.nn.Parameter(torch.rand(len(qual_levels[i]), latent_dim)))

        self.X_qual_bounds = torch.zeros(2, self.num_latent)
        self.X_qual_bounds[0, :] = -10
        self.X_qual_bounds[1, :] = 10

    def convert_qual_to_latent(self, X_qual: Tensor):
        assert X_qual.shape[1] == self.num_qual, f"X_qual has {X_qual.shape[1]} features, but expected {self.num_qual}."
        latents = None
        for k in range(self.num_qual):
            map_k = self.latent_params[k]
            num_feats = map_k.shape[1]

            curr_latents = torch.zeros((X_qual.shape[0], num_feats))

            for i, level in enumerate(self.qual_levels[k]):
                mask = X_qual == level
                curr_latents[mask.squeeze(1)] = map_k[i]

            if latents is None:
                latents = curr_latents
            else:
                latents = torch.cat((latents, curr_latents), dim=1)

        return latents


class LVKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_cont: int, num_qual: int, qual_levels: list, **kwargs):
        """
        Initialize latent variable kernel. During initializiation determines how
        latent vector size based on number of levels. If nummber of levels > 2
        then latent vector is length 2 and 1 otherwise.

        Note:
        Level 1 should be initialized to 0 vector and not optimized under MLE.
        Level 2 first index should be 0 and not optimized as well.
        :param num_cont: Number of continuous variables
        :param num_qual: Number of qualitative variables
        :param qual_levels: List of lists containing levels for qualitative levels
        should be indexed the same as data when passed through forward()
        :param kwargs: Additional Gpytorch kernel params
        """
        super().__init__(**kwargs)
        self.num_cont = num_cont
        self.latent_feat_nums = list()

        # Register continuous parameters for MLE
        self.register_parameter(name='quant_params',
                                parameter=torch.nn.Parameter(torch.rand(num_cont)))

        self.num_latent = 0
        self.latent_params = list()
        self.latent_feat_nums = list()
        self.num_qual = num_qual
        self.qual_levels = qual_levels
        self.total_features = self.num_qual + self.num_cont

        for i in range(num_qual):
            # TODO: Don't optimize level 1 and level 2 index 0
            latent_dim = 2 if len(qual_levels[i]) > 2 else 1
            self.num_latent += latent_dim
            self.latent_feat_nums.append(latent_dim)
            self.register_parameter(name=f"latent_param_{i}",
                                    parameter=torch.nn.Parameter(torch.rand(len(qual_levels[i]), latent_dim)))

        self.X_qual_bounds = torch.zeros(2, 2)
        self.X_qual_bounds[0, :] = -10
        self.X_qual_bounds[1, :] = 10
        self.X_qual_bounds = self.X_qual_bounds.repeat(1, self.num_qual)

    def continous_correlation(self, x1: Tensor, x2: Tensor):
        squared_diff = (x1 - x2) ** 2
        return -1 * torch.sum(self.quant_params * squared_diff)

    def qualitative_correlation(self, x1: Tensor, x2: Tensor):
        return -1 * sum(torch.sub(x1, x2).pow(2))

    def convert_qual_to_latent(self, X_qual: Tensor):
        latents = None
        for k in range(self.num_qual):
            map_k = getattr(self, f"latent_param_{k}")
            num_feats = map_k.shape[1]

            curr_latents = torch.zeros((X_qual.shape[0], num_feats))

            for i, level in enumerate(self.qual_levels[k]):
                mask = X_qual == level
                curr_latents[mask.squeeze(1)] = map_k[i]

            if latents is None:
                latents = curr_latents
            else:
                latents = torch.cat((latents, curr_latents), dim=1)

        return latents

    def LV_dist(self, x1: Tensor, x2: Tensor):
        """
        Calculated squared difference of each continuous feature multiplied by that feature's learned weight.
        Calculated squared-L2 norm of latent vector mappings of qualitative variables and sum with continous
        variable kernel values.
        :param x1: Tensor of design of experiment samples
        :param x2: Second tensor of design of experiment samples
        :return: Covariance matrix
        """
        x1 = x1[0]
        x2 = x2[0]
        N = x1.shape[0]
        M = x2.shape[0]
        cov = torch.zeros(N, M)
        x1_latent = self.convert_qual_to_latent(x1[:, self.num_cont:].type(torch.LongTensor))
        x2_latent = self.convert_qual_to_latent(x2[:, self.num_cont:].type(torch.LongTensor))

        x1 = torch.cat((x1[:, :self.num_cont], x1_latent), dim=1)
        x2 = torch.cat((x2[:, :self.num_cont], x2_latent), dim=1)

        for i in range(N):
            for j in range(M):
                cov[i, j] = self.continous_correlation(x1=x1[i, :self.num_cont], x2=x2[j, :self.num_cont])

                qual_sum = 0
                curr_idx = 0
                x1_latent = x1[i, self.num_cont:]
                x2_latent = x2[j, self.num_cont:]
                for num_feats in self.latent_feat_nums:
                    end_idx = curr_idx + num_feats
                    qual_sum += self.qualitative_correlation(x1_latent[curr_idx:end_idx], x2_latent[curr_idx:end_idx])
                    curr_idx = end_idx

                cov[i, j] += qual_sum

        return torch.exp(cov).unsqueeze(0)

    def forward(self, x1: Tensor, x2: Tensor, **kwargs):
        # assert x1.shape[
        #            1] == self.total_features, f"x1 has {x1.shape[1]} features, expected {self.total_features}"
        # assert x2.shape[
        #            1] == self.total_features, f"x2 has {x2.shape[1]} features, expected {self.total_features}"
        return self.LV_dist(x1, x2)
