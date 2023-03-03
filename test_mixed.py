import torch
import numpy as np
from test_functions.Branin import Branin
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

BATCH_SIZE = 3
d = 2
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_function = Branin()


def gen_initial_data(n=5):
    # generate training data
    train_x = torch.linspace(0, 1, n).unsqueeze(-1)
    sampled_quals = torch.Tensor(np.random.choice(test_function.qual_levels[0], n)).unsqueeze(-1)
    train_obj = test_function.forward(X_cont=train_x, X_qual=sampled_quals)  # add output dimension
    best_observed_value = train_obj.max().item()
    train_x = torch.column_stack((train_x, sampled_quals))
    return train_x, train_obj, best_observed_value


def get_fitted_model(train_x, train_obj, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_model(mll)
    return model


def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([
            torch.zeros(d, dtype=dtype, device=device),
            torch.ones(d, dtype=dtype, device=device),
        ]),
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=200,
    )

    # observe new values
    new_x = candidates.detach()
    x_qual_new_copy = torch.clone(new_x[:, 1])
    for level in test_function.qual_levels[0]:
        mask = (0.333 / 2) >= torch.abs(x_qual_new_copy - level)
        new_x[mask, 1] = level

    new_obj = test_function.forward(new_x[:, 0], new_x[:, 1])  # add output dimension
    return new_x, new_obj


seed = 1
torch.manual_seed(seed)

N_BATCH = 100
best_observed = []

# call helper function to initialize model
train_x, train_obj, best_value = gen_initial_data(n=5)
best_observed.append(best_value)

state_dict = None
# run N_BATCH rounds of BayesOpt after the initial random batch
for iteration in range(N_BATCH):
    # fit the model
    model = get_fitted_model(
        train_x,
        standardize(train_obj),
        state_dict=state_dict,
    )

    # define the qNEI acquisition module using a QMC sampler
    qEI = qExpectedImprovement(model=model, best_f=standardize(train_obj).max())

    # optimize and get new observation
    new_x, new_obj = optimize_acqf_and_get_observation(qEI)

    # update training points
    train_x = torch.cat((train_x, new_x))
    train_obj = torch.cat((train_obj, new_obj.unsqueeze(-1)))

    # update progress
    best_value = train_obj.max().item()
    best_observed.append(best_value)

    state_dict = model.state_dict()

    print(train_obj.max())
