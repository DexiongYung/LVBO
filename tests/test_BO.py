import torch
from utilities.test_utilities import test_func_dict
from utilities.test_utilities import sample_DOE
from algorithms.factory import alg_factory
from tests.default_test_args import create_default_args


def run(args):
    test_function = test_func_dict[args.function]
    X_cont, X_qual, Y = sample_DOE(name_func=args.function, test_func=test_function)
    BO_alg = alg_factory[args.algorithm](X_cont=X_cont, X_qual=X_qual, Y=Y,
                                         X_cont_bounds=test_function.X_cont_bounds_tnsr,
                                         X_qual_bounds=torch.Tensor([[-0.15], [1.15]]),
                                         X_qual_levels=test_function.qual_levels)

    for i in range(50):
        candidates = BO_alg.run_iteration_of_BO()
        Y = test_function.forward(X_cont=candidates[:, 0], X_qual=candidates[:, 1])
        BO_alg.add_to_DOE(candidates=candidates, Y=Y)
        print(BO_alg.Y.max())


if __name__ == "__main__":
    parser = create_default_args()
    run(parser.parse_args())
