
import os
import torch
from botorch.test_functions import StyblinskiTang
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
from matplotlib import pyplot as plt
import time
import warnings

def inputs():

    all_train_x_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x.pt'
    all_train_obj_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj.pt'

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    SMOKE_TEST = os.environ.get("SMOKE_TEST")

    synthetic_test_function = StyblinskiTang(negate=True)
    dimensions = 2

    ##### Remember to change this!!!!!!
    bounds = torch.tensor([[-5] * dimensions, [5] * dimensions], device=device, dtype=dtype)
    ### !!!!!!!!!!!!!!!!!!

    NOISE_SE = 0

    initial_data_points = 12

    BATCH_SIZE = 12 if not SMOKE_TEST else 2
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 32
    MC_SAMPLES = 256 if not SMOKE_TEST else 32

    return device, dtype, SMOKE_TEST, bounds, NOISE_SE, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, initial_data_points, dimensions, synthetic_test_function, all_train_x_location, all_train_obj_location

def outcome_constraint(X):
    """L1 constraint; feasible if less than or equal to zero."""
    return X.sum(dim=-1) - 3

def weighted_obj(synthetic_test_function, X):
    """Feasibility weighted objective; zero if not feasible."""
    return synthetic_test_function(X) * (outcome_constraint(X) <= 0).type_as(X)

def get_data(train_x_location, train_obj_location):
    train_x = torch.load(train_x_location)
    train_obj = torch.load(train_obj_location)
    train_obj = torch.reshape(train_obj, (len(train_obj), 1))

    return train_x, train_obj

def generate_initial_data(synthetic_test_function, train_x_location, train_obj_location):
    # generate training data
    train_x, train_obj = get_data(train_x_location, train_obj_location)
    train_con = outcome_constraint(train_x).unsqueeze(-1)  # add output dimension
    best_observed_value = weighted_obj(synthetic_test_function, train_x).max().item()
    print("best observed value: ", best_observed_value)

    return train_x, train_obj, train_con, best_observed_value

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

def obj_callable(Z):
    return Z[..., 0]

def constraint_callable(Z):
    return Z[..., 1]

def optimize_acqf_and_get_observation(acq_func, bounds, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, synthetic_test_function, NOISE_SE):
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
    exact_obj = synthetic_test_function(new_x).unsqueeze(-1)  # add output dimension
    exact_con = outcome_constraint(new_x).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)

    return new_x, new_obj, new_con

def update_random_observations(synthetic_test_function, best_random, BATCH_SIZE):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = torch.rand(BATCH_SIZE, 6)
    next_random_best = weighted_obj(synthetic_test_function, rand_x).max().item()
    best_random.append(max(best_random[-1], next_random_best))

    return best_random

def plot(synthetic_test_function, BATCH_SIZE, best_observed_all_nei):
    GLOBAL_MAXIMUM = synthetic_test_function.optimal_value

    iters = np.arange(1 + 1) * BATCH_SIZE
    y_nei = np.asarray(best_observed_all_nei)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(iters, y_nei.mean(axis=0), label="qNEI", linewidth=1.5)
    plt.plot(
        [0, BATCH_SIZE],
        [GLOBAL_MAXIMUM] * 2,
        "k",
        label="true best objective",
        linewidth=2,
    )
    ax.set_ylim(bottom=0.5)
    ax.set(
        xlabel="number of observations (beyond initial points)",
        ylabel="best objective value",
    )
    ax.legend(loc="lower right")

    plt.show()

def plot_scatter(train_x_nei, train_obj_nei):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.scatter(train_x_nei[:, 0], train_x_nei[:, 1], c=train_obj_nei, cmap='viridis_r')
    #ax.invert_yaxis()

    plt.show()

def main():
    device, dtype, SMOKE_TEST, bounds, NOISE_SE, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, MC_SAMPLES, initial_data_points, dimensions, synthetic_test_function, train_x_location, train_obj_location = inputs()

    train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable]
    )

    # ### Perform Bayesian Optimization loop with qNEI
    # The Bayesian optimization "loop" for a batch size of $q$ simply iterates the following steps:
    # 1. given a surrogate model, choose a batch of points $\{x_1, x_2, \ldots x_q\}$
    # 2. observe $f(x)$ for each $x$ in the batch
    # 3. update the surrogate model.

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    best_observed_all_nei = []
    best_observed_nei = []

    # call helper functions to generate initial training data and initialize model
    (
        train_x_nei,
        train_obj_nei,
        train_con_nei,
        best_observed_value_nei,
    ) = generate_initial_data(synthetic_test_function, train_x_location, train_obj_location)
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei, train_yvar)

    best_observed_nei.append(best_observed_value_nei)

    t0 = time.monotonic()

    # fit the models
    fit_gpytorch_mll(mll_nei)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # for best_f, we use the best observed noisy values as an approximation
    qNEI = qNoisyExpectedImprovement(
        model=model_nei,
        X_baseline=train_x_nei,
        sampler=qmc_sampler,
        objective=constrained_obj,
    )

    # optimize and get new observation
    new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(qNEI, bounds, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, synthetic_test_function, NOISE_SE)

    # update training points
    train_x_nei = torch.cat([train_x_nei, new_x_nei])
    train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
    train_con_nei = torch.cat([train_con_nei, new_con_nei])

    # update progress
    best_value_nei = weighted_obj(synthetic_test_function, train_x_nei).max().item()
    best_observed_nei.append(best_value_nei)

    t1 = time.monotonic()

    print(
        f"time = {t1-t0:>4.2f}.",
        end="",
    )

    best_observed_all_nei.append(best_observed_nei)

    plot(synthetic_test_function, BATCH_SIZE, best_observed_all_nei)
    plot_scatter(train_x_nei, train_obj_nei)

    all_train_x_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x.pt'
    all_train_obj_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj.pt'

    torch.save(train_x_nei, all_train_x_location)
    torch.save(train_obj_nei, all_train_obj_location)

main()