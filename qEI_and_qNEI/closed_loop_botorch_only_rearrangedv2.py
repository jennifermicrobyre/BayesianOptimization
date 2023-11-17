
import os
import torch
from botorch.test_functions import Hartmann
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
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    SMOKE_TEST = os.environ.get("SMOKE_TEST")

    synthetic_test_function = StyblinskiTang(negate=True)
    dimensions = 2
    bounds = torch.tensor([[-5] * dimensions, [5] * dimensions], device=device, dtype=dtype)

    NOISE_SE = 0

    initial_data_points = 12

    BATCH_SIZE = 1 if not SMOKE_TEST else 2
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 32

    N_TRIALS = 10 if not SMOKE_TEST else 2
    N_BATCH = 100 if not SMOKE_TEST else 2
    MC_SAMPLES = 256 if not SMOKE_TEST else 32

    return device, dtype, SMOKE_TEST, bounds, NOISE_SE, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, N_TRIALS, N_BATCH, MC_SAMPLES, initial_data_points, dimensions, synthetic_test_function

def outcome_constraint(X):
    """L1 constraint; feasible if less than or equal to zero."""
    return X.sum(dim=-1) - 3

def weighted_obj(synthetic_test_function, X):
    """Feasibility weighted objective; zero if not feasible."""
    return synthetic_test_function(X) * (outcome_constraint(X) <= 0).type_as(X)

def generate_initial_data(device, dtype, synthetic_test_function, NOISE_SE, initial_data_points, dimensions):
    # generate training data
    train_x = torch.rand(initial_data_points, dimensions, device=device, dtype=dtype)
    exact_obj = synthetic_test_function(train_x).unsqueeze(-1)  # add output dimension
    exact_con = outcome_constraint(train_x).unsqueeze(-1)  # add output dimension
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    best_observed_value = weighted_obj(synthetic_test_function, train_x).max().item()
    return train_x, train_obj, train_con, best_observed_value

def initialize_model(train_x, train_obj, train_con, train_yvar, state_dict=None):
    # define models for objective and constraint
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(
        train_x
    )
    model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(
        train_x
    )
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
        options={"batch_limit": 5, "maxiter": 200},
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

def ci(y, N_TRIALS):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

def plot(synthetic_test_function, N_BATCH, BATCH_SIZE, N_TRIALS, best_observed_all_ei, best_observed_all_nei, best_random_all):
    GLOBAL_MAXIMUM = synthetic_test_function.optimal_value

    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    y_ei = np.asarray(best_observed_all_ei)
    y_nei = np.asarray(best_observed_all_nei)
    y_rnd = np.asarray(best_random_all)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(iters, y_rnd.mean(axis=0), yerr=ci(y_rnd, N_TRIALS), label="random", linewidth=1.5)
    ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei, N_TRIALS), label="qEI", linewidth=1.5)
    ax.errorbar(iters, y_nei.mean(axis=0), yerr=ci(y_nei, N_TRIALS), label="qNEI", linewidth=1.5)
    plt.plot(
        [0, N_BATCH * BATCH_SIZE],
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
    device, dtype, SMOKE_TEST, bounds, NOISE_SE, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, N_TRIALS, N_BATCH, MC_SAMPLES, initial_data_points, dimensions, synthetic_test_function = inputs()

    verbose = True

    train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable],
    )

    # ### Perform Bayesian Optimization loop with qNEI
    # The Bayesian optimization "loop" for a batch size of $q$ simply iterates the following steps:
    # 1. given a surrogate model, choose a batch of points $\{x_1, x_2, \ldots x_q\}$
    # 2. observe $f(x)$ for each $x$ in the batch
    # 3. update the surrogate model.

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    best_observed_all_ei, best_observed_all_nei, best_random_all = [], [], []

    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_nei, best_random = [], [], []

        # call helper functions to generate initial training data and initialize model
        (
            train_x_ei,
            train_obj_ei,
            train_con_ei,
            best_observed_value_ei,
        ) = generate_initial_data(device, dtype, synthetic_test_function, NOISE_SE, initial_data_points, dimensions)
        mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei, train_con_ei, train_yvar)

        train_x_nei, train_obj_nei, train_con_nei = train_x_ei, train_obj_ei, train_con_ei
        best_observed_value_nei = best_observed_value_ei
        mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei, train_yvar)

        best_observed_ei.append(best_observed_value_ei)
        best_observed_nei.append(best_observed_value_nei)
        best_random.append(best_observed_value_ei)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            t0 = time.monotonic()

            # fit the models
            fit_gpytorch_mll(mll_ei)
            fit_gpytorch_mll(mll_nei)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

            # for best_f, we use the best observed noisy values as an approximation
            qEI = qExpectedImprovement(
                model=model_ei,
                best_f=(train_obj_ei * (train_con_ei <= 0).to(train_obj_ei)).max(),
                sampler=qmc_sampler,
                objective=constrained_obj,
            )

            qNEI = qNoisyExpectedImprovement(
                model=model_nei,
                X_baseline=train_x_nei,
                sampler=qmc_sampler,
                objective=constrained_obj,
            )

            # optimize and get new observation
            new_x_ei, new_obj_ei, new_con_ei = optimize_acqf_and_get_observation(qEI, bounds, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, synthetic_test_function, NOISE_SE)
            new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(qNEI, bounds, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, synthetic_test_function, NOISE_SE)

            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
            train_con_ei = torch.cat([train_con_ei, new_con_ei])

            train_x_nei = torch.cat([train_x_nei, new_x_nei])
            train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
            train_con_nei = torch.cat([train_con_nei, new_con_nei])

            #JH
            if trial == 1 and iteration == 1:
                train_x_nei_all = new_x_nei
                train_obj_nei_all = new_obj_nei
            else:
                train_x_nei_all = torch.cat([train_x_nei_all, new_x_nei])
                train_obj_nei_all = torch.cat([train_obj_nei_all, new_obj_nei])

            #print(train_x_nei)
            #print(train_obj_nei)

            #plot_scatter(train_x_nei, train_obj_nei)

            # update progress
            best_random = update_random_observations(synthetic_test_function, best_random, BATCH_SIZE)
            best_value_ei = weighted_obj(synthetic_test_function, train_x_ei).max().item()
            best_value_nei = weighted_obj(synthetic_test_function, train_x_nei).max().item()
            best_observed_ei.append(best_value_ei)
            best_observed_nei.append(best_value_nei)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei,
                train_obj_ei,
                train_con_ei,
                train_yvar,
                model_ei.state_dict(),
            )
            mll_nei, model_nei = initialize_model(
                train_x_nei,
                train_obj_nei,
                train_con_nei,
                train_yvar,
                model_nei.state_dict(),
            )

            t1 = time.monotonic()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI) = "
                    f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")

        best_observed_all_ei.append(best_observed_ei)
        best_observed_all_nei.append(best_observed_nei)
        best_random_all.append(best_random)


        plot(synthetic_test_function, N_BATCH, BATCH_SIZE, N_TRIALS, best_observed_all_ei, best_observed_all_nei, best_random_all)
    plot_scatter(train_x_nei_all, train_obj_nei_all)

main()