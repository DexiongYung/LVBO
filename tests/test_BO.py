from utilities.test_utilities import test_func_dict
from utilities.test_utilities import sample_DOE
from algorithms.factory import alg_factory
from tests.default_test_args import create_default_args


def run(args):
    test_function = test_func_dict[args.function]
    X_cont, X_qual, Y = sample_DOE(name_func=args.function, test_func=test_function)
    BO_alg = alg_factory[args.algorithm](X_cont=X_cont, X_qual=X_qual, Y=Y, qual_levels=test_function.qual_levels,
                                         device="cpu", X_cont_bounds=test_function.X_cont_bounds_tnsr)
    BO_alg.fit_mll()
    BO_alg.optimize_acq()


if __name__ == "__main__":
    parser = create_default_args()
    run(parser.parse_args())
