import os
import pandas as pd
import torch
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import numpy as np
import time
import warnings

def inputs():
    train_obj_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj.csv'
    train_obj_all_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj_all.pt'

    train_x_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x.pt'
    train_x_all_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x_all.pt'

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    SMOKE_TEST = os.environ.get("SMOKE_TEST")

    dimensions = 2
    ##### Remember to change this!!!!!!
    bounds = torch.tensor([[-5] * dimensions, [5] * dimensions], device=device, dtype=dtype)
    if torch.cuda.is_available():
        bounds = bounds.to('cuda')
    ### !!!!!!!!!!!!!!!!!!

    NOISE_SE = 0

    initial_data_points = 12

    BATCH_SIZE = 12 if not SMOKE_TEST else 2
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 32
    MC_SAMPLES = 256 if not SMOKE_TEST else 32

    return device, dtype, SMOKE_TEST, bounds, NOISE_SE, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, initial_data_points, dimensions, train_obj_location, train_obj_all_location, train_x_location, train_x_all_location

def load_obj_make_tensor_and_save(train_obj_location, train_obj_all_location):
    train_obj = np.loadtxt(train_obj_location, delimiter=",")
    train_obj = torch.from_numpy(train_obj)
    if torch.cuda.is_available():
        train_obj = train_obj.to('cuda')

    train_obj = torch.reshape(train_obj, (len(train_obj), 1))

    train_obj_all = torch.load(train_obj_all_location)
    train_obj_all = torch.cat([train_obj_all, train_obj])

    torch.save(train_obj, train_obj_location[:-3] + 'pt')
    torch.save(train_obj_all, train_obj_all_location)

    return train_obj, train_obj_all

def load_train_x_and_train_x_all(train_x_location, train_x_all_location):
    train_x = torch.load(train_x_location)
    train_x_all = torch.load(train_x_all_location)
    if torch.cuda.is_available():
        train_x = train_x.to('cuda')
        train_x_all = train_x_all.to('cuda')

    return train_x, train_x_all

def obj_callable(Z):
    return Z[..., 0]

def constraint_callable(Z):
    return Z[..., 1]

def outcome_constraint(X):
    """L1 constraint; feasible if less than or equal to zero."""
    return X.sum(dim=-1) - 3

def initialize_model(train_x, train_obj, train_con, train_yvar, state_dict=None):

    # define models for objective and constraint
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(train_x)

    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model

def optimize_acqf_and_get_new_values_to_sample(acq_func, bounds, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": BATCH_SIZE, "maxiter": 2000},
    )
    # observe new values
    new_x = candidates.detach()

    return new_x

def save_as_csv(new_train_x, train_x_location):
    new_train_x_np = new_train_x.numpy()
    new_train_x_df = pd.DataFrame(new_train_x_np)
    new_train_x_df.to_csv(train_x_location[:-2] + 'csv')

def main():
    device, dtype, SMOKE_TEST, bounds, NOISE_SE, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, initial_data_points, dimensions, train_obj_location, train_obj_all_location, train_x_location, train_x_all_location = inputs()
    train_obj, train_obj_all = load_obj_make_tensor_and_save(train_obj_location, train_obj_all_location)
    train_x, train_x_all = load_train_x_and_train_x_all(train_x_location, train_x_all_location)
    train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)

    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(objective=obj_callable, constraints=[constraint_callable])

    # ### Perform Bayesian Optimization loop with qNEI
    # The Bayesian optimization "loop" for a batch size of $q$ simply iterates the following steps:
    # 1. given a surrogate model, choose a batch of points $\{x_1, x_2, \ldots x_q\}$
    # 2. observe $f(x)$ for each $x$ in the batch
    # 3. update the surrogate model.

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    train_con_all = outcome_constraint(train_x_all).unsqueeze(-1)  # add output dimension
    mll_nei, model_nei = initialize_model(train_x_all, train_obj_all, train_con_all, train_yvar)

    t0 = time.monotonic()

    # fit the models
    fit_gpytorch_mll(mll_nei)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # for best_f, we use the best observed noisy values as an approximation
    qNEI = qNoisyExpectedImprovement(
        model=model_nei,
        X_baseline=train_x_all,
        sampler=qmc_sampler,
        objective=constrained_obj,
    )

    new_train_x = optimize_acqf_and_get_new_values_to_sample(qNEI, bounds, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES)

    # save new training points
    torch.save(new_train_x, train_x_location)
    save_as_csv(new_train_x, train_x_location)

    # update training points
    train_x_all = torch.cat([train_x_all, new_train_x])
    torch.save(train_x_all, train_x_all_location)

    t1 = time.monotonic()

    print(f"time = {t1 - t0:>4.2f}.", end="", )


main()