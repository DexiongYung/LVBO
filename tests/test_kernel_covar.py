import torch
import seaborn as sns
import matplotlib.pyplot as plt

from utilities.test_utilities import test_func_dict, sample_DOE
from utilities.tensor_utilities import is_positive_definite
from tests.default_test_args import create_default_args
from kernels.LVKernel import LVKernel

kernel_dict = dict(LV=LVKernel)


def run(args):
    test_fn = test_func_dict[args.function]
    kernel_fn = kernel_dict[args.kernel]
    X_cont, X_qual, _ = sample_DOE(name_func=args.function, test_func=test_fn)
    kernel = kernel_fn(num_cont=1, num_qual=1, qual_levels=test_fn.qual_levels)
    X_latent = kernel.convert_qual_to_latent(X_qual=X_qual)
    X = torch.column_stack((X_cont, X_latent))
    covar_matrix = kernel.forward(x1=X, x2=X)
    sns.heatmap(covar_matrix.detach().numpy())
    print(f'is positive definite? {is_positive_definite(covar_matrix)}')
    plt.show()


if __name__ == "__main__":
    parser = create_default_args()
    run(parser.parse_args())
