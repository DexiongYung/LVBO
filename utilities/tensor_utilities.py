import torch
from torch import Tensor


def is_positive_definite(mat: Tensor):
    return bool((mat == mat.T).all() and (torch.eig(mat)[0][:, 0] >= 0).all())


def qual_to_integers(X_qual: Tensor, qual_levels: list):
    M = len(qual_levels)
    X_qual_copy = X_qual.clone().detach()

    for i in range(M):
        N = len(qual_levels[i])
        for j in range(N):
            value = qual_levels[i][j]
            mask = X_qual[:, i] == value
            X_qual_copy[mask] = j
    X_qual_copy = X_qual_copy.type(torch.LongTensor)
    return X_qual_copy


def is_tensor_in_qual_levels(X_qual: Tensor, qual_levels: list):
    num_q = len(qual_levels)

    for i in range(num_q):
        if not all(torch.isin(X_qual[:, i], torch.Tensor(qual_levels[i]))):
            return False

    return True


def is_tensor_in_cont_bounds(X_cont: Tensor, cont_bounds: list):
    num_c = len(cont_bounds)

    for i in range(num_c):
        vals = X_cont[:, i]
        lower_b = cont_bounds[i][0]
        upper_b = cont_bounds[i][1]
        is_valid = all(lower_b <= vals) and all(vals <= upper_b)

        if not is_valid:
            return False

    return True
