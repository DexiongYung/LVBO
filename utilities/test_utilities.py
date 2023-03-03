import torch
import numpy as np
from test_functions.Branin import Branin

test_func_dict = dict(Branin=Branin())


def sample_DOE(name_func: str, test_func):
    if name_func == "Branin":
        sample_num = 5
        X_cont = torch.linspace(0, 1, sample_num).unsqueeze(1)
        X_qual = torch.Tensor(np.random.choice(test_func.qual_levels[0], sample_num)).unsqueeze(1)
        Y = torch.Tensor(test_func.forward(X_cont=X_cont, X_qual=X_qual))

    return X_cont, X_qual, Y
