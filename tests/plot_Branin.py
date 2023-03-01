import torch
import numpy as np
import matplotlib.pyplot as plt
from test_functions.Branin import Branin
from tests.default_test_args import create_default_args


def run(args):
    test_func = Branin()
    X_cont = torch.Tensor(np.array(args.continuous.split(",")).astype(float)).unsqueeze(0)
    X_qual = torch.Tensor(np.array(args.qualitative.split(",")).astype(float)).unsqueeze(0)
    out = test_func.forward(X_cont=X_cont, X_qual=X_qual)

    plt.plot(X_cont.numpy()[0], out.numpy()[0])
    plt.show()


if __name__ == "__main__":
    parser = create_default_args()
    parser.add_argument("-c", "--continuous", type=str, help="Continuous value to input into function",
                        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("-q", "--qualitative", type=str, help="Qualitative value to input",
                        default="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0")
    run(parser.parse_args())
