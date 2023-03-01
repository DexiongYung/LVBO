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

        self.X_latent_bounds = torch.zeros(2, self.num_latent)
        self.X_latent_bounds[0, :] = -10
        self.X_latent_bounds[1, :] = 10

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
    def __init__(self, num_cont: int, latent_feat_nums: list, **kwargs):
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
        self.latent_feat_nums = latent_feat_nums
        self.total_features = self.num_cont + sum(self.latent_feat_nums)

        # Register continuous parameters for MLE
        self.register_parameter(name='quant_params',
                                parameter=torch.nn.Parameter(torch.rand(num_cont)))



    def continous_correlation(self, x1: Tensor, x2: Tensor):
        squared_diff = (x1 - x2) ** 2
        return -1 * torch.sum(self.quant_params * squared_diff)

    def qualitative_correlation(self, x1: Tensor, x2: Tensor):
        return -1 * sum(torch.sub(x1, x2).pow(2))

    def LV_dist(self, x1: Tensor, x2: Tensor):
        """
        Calculated squared difference of each continuous feature multiplied by that feature's learned weight.
        Calculated squared-L2 norm of latent vector mappings of qualitative variables and sum with continous
        variable kernel values.
        :param x1: Tensor of design of experiment samples
        :param x2: Second tensor of design of experiment samples
        :return: Covariance matrix
        """
        N = x1.shape[0]
        M = x2.shape[0]
        cov = torch.zeros(N, M)

        for i in range(N):
            for j in range(M):
                cov[i, j] = self.continous_correlation(x1=x1[i, :self.num_cont], x2=x2[j, :self.num_cont])

                qual_sum = 0
                x1_qual = x1[i, self.num_cont:].type(torch.LongTensor)
                x2_qual = x2[j, self.num_cont:].type(torch.LongTensor)
                curr_idx = 0
                for num_feats in self.latent_feat_nums:
                    end_idx = curr_idx + num_feats
                    qual_sum += self.qualitative_correlation(x1_qual[curr_idx:end_idx], x2_qual[curr_idx:end_idx])
                    curr_idx = end_idx

                cov[i, j] += qual_sum

        return torch.exp(cov)

    def forward(self, x1: Tensor, x2: Tensor, **kwargs):
        assert x1.shape[1] == self.total_features, f"x1 has {x1.shape[1]} features, expected {self.total_features}"
        assert x2.shape[1] == self.total_features, f"x2 has {x2.shape[1]} features, expected {self.total_features}"
        return self.LV_dist(x1, x2)
