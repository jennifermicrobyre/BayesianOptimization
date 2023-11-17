import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
import time
import warnings
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

def generate_initial_data(n=32):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)
    problem = BraninCurrin(negate=True).to(**tkwargs)
    print("problem.bounds: ", problem.bounds)
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE

    return train_x, train_obj, train_obj_true

def initialize_model(train_x, train_obj, bounds, NOISE_SE):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # define models for objective and constraint
    train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return mll, model

def optimize_qnehvi_and_get_observation(model, train_x, sampler, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)
    problem = BraninCurrin(negate=True).to(**tkwargs)

    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1

    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        X_baseline=normalize(train_x, problem.bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE

    return new_x, new_obj, new_obj_true

def plot_outputs(N_BATCH, BATCH_SIZE, train_obj_true_qnehvi):
    fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True, sharey=True)
    algos = ["qNEHVI"]
    cm = plt.cm.get_cmap('viridis')

    #batch_number = torch.cat(
    #    [torch.zeros(2 * (problem.dim + 1)), torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
    #).numpy()
    batch_number = torch.cat(
        [torch.zeros(32), torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
    ).numpy()
    axes.scatter(train_obj_true_qnehvi[:, 0].cpu().numpy(), train_obj_true_qnehvi[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)
    axes.set_title("qNEHVI")
    axes.set_xlabel("Objective 1")
    #axes.set_xlim(-150, 5)
    #axes.set_ylim(-15, 0)
    axes.set_ylabel("Objective 2")
    norm = plt.Normalize(batch_number.min(), batch_number.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title("Iteration")

    plt.show()

def plot_inputs(N_BATCH, BATCH_SIZE, train_x_qnehvi):
    fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True, sharey=True)
    algos = ["qNEHVI"]
    cm = plt.cm.get_cmap('viridis')

    #batch_number = torch.cat(
    #    [torch.zeros(2 * (problem.dim + 1)), torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
    #).numpy()
    batch_number = torch.cat(
        [torch.zeros(32), torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1)]
    ).numpy()
    axes.scatter(train_x_qnehvi[:, 0].cpu().numpy(), train_x_qnehvi[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)
    axes.set_title("qNEHVI")
    axes.set_xlabel("Input 1")
    #axes.set_xlim(-150, 5)
    #axes.set_ylim(-15, 0)
    axes.set_ylabel("Input 2")
    norm = plt.Normalize(batch_number.min(), batch_number.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title("Iteration")

    plt.show()

def plot_log_hypervolume_difference(N_BATCH, BATCH_SIZE, hvs_qnehvi):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    problem = BraninCurrin(negate=True).to(**tkwargs)

    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    log_hv_difference_qnehvi = np.log10(problem.max_hv - np.asarray(hvs_qnehvi))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        iters, log_hv_difference_qnehvi, label="qNEHVI", linewidth=1.5,
    )
    ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log Hypervolume Difference')
    ax.legend(loc="lower left")
    plt.show()

def main():
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    SMOKE_TEST = os.environ.get("SMOKE_TEST")

    problem = BraninCurrin(negate=True).to(**tkwargs)
    bounds = problem.bounds
    print(f"Device tensor is stored on: {bounds.device}")
    problem_dimensions = problem.dim

    NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)

    BATCH_SIZE = 44
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4

    standard_bounds = torch.zeros(2, problem_dimensions, **tkwargs)
    standard_bounds[1] = 1

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    N_BATCH = 2 if not SMOKE_TEST else 10
    MC_SAMPLES = 128 if not SMOKE_TEST else 16

    verbose = True

    # call helper functions to generate initial training data and initialize model
    #train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = generate_initial_data(
    #    n=2 * (problem.dim + 1)
    #)
    train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = generate_initial_data(n=32)

    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, bounds, NOISE_SE)

    # compute hypervolume
    #hvs_qnehvi = []
    #bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true_qnehvi)
    #volume = bd.compute_hypervolume().item()
    #hvs_qnehvi.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):
        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_qnehvi)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # optimize acquisition functions and get new observations
        new_x_qnehvi, new_obj_qnehvi, new_obj_true_qnehvi = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x_qnehvi, qnehvi_sampler, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES)

        train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
        train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
        train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

        # update progress
        #for hvs_list, train_obj in zip(
        #        (hvs_qnehvi),
        #        (
        #                train_obj_true_qnehvi,
        #        ),
        #):
            # compute hypervolume
        #    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
        #    volume = bd.compute_hypervolume().item()
        #    hvs_qnehvi.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi, bounds, NOISE_SE)

        t1 = time.monotonic()

        #if verbose:
        #    print(
        #        f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
        #        f"({hvs_qnehvi[-1]:>4.2f}), "
        #        f"time = {t1 - t0:>4.2f}.",
        #        end="",
        #    )
        #else:
        #    print(".", end="")

        if verbose:
            print(
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

    #### plot stuff
    #plot_log_hypervolume_difference(N_BATCH, BATCH_SIZE, hvs_qnehvi)
    plot_outputs(N_BATCH, BATCH_SIZE, train_obj_true_qnehvi)
    plot_inputs(N_BATCH, BATCH_SIZE, train_x_qnehvi)


main()