import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.test_functions.multi_objective import BraninCurrin
import time
import warnings
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

def inputs():
    num_samples = 1
    number_of_repeats = 45
    number_of_cycles = 45

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    bounds = torch.tensor([[0, 0], [1, 1]], **tkwargs)
    noise_se_outputs = torch.tensor([5, 0.5], **tkwargs)
    noise_se_inputs = torch.tensor([0.05, 0.05], **tkwargs)

    num_restarts = 10
    raw_samples = 512
    mc_samples = 128

    problem_dimensions = 2
    ref_point = [-18.0, -6.0]

    standard_bounds = torch.zeros(2, problem_dimensions, dtype=torch.float64)
    standard_bounds[1] = 1

    save_x_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x.pt'
    all_x_save_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x_all.pt'

    all_true_outputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all_true.pt'
    all_outputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all.pt'

    return num_samples, number_of_repeats, number_of_cycles, bounds, noise_se_outputs, noise_se_inputs, num_restarts, \
           raw_samples, mc_samples, problem_dimensions, ref_point, standard_bounds, save_x_location, all_x_save_location, all_true_outputs_location, all_outputs_location

def generate_initial_inputs(bounds, num_samples, save_x_location, all_x_save_location):
    train_x = draw_sobol_samples(bounds=bounds, n=num_samples, q=1).squeeze(1)
    torch.save(train_x, save_x_location)
    torch.save(train_x, all_x_save_location)

def duplicate(train_x, number_of_repeats):
    duplicated_train_x = torch.cat((train_x, train_x), 0)
    for i in range(number_of_repeats - 2):
        duplicated_train_x = torch.cat((duplicated_train_x, train_x), 0)

    return duplicated_train_x

def get_noisy_train_x(train_x, noise_se_inputs):
    noisy_train_x = train_x + torch.randn_like(train_x) * noise_se_inputs
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            if noisy_train_x[i, j] < 0 or noisy_train_x[i, j] > 1:
                noisy_train_x[i, j] = train_x[i, j]

    return noisy_train_x

def get_outputs(train_x, number_of_samples, number_of_repeats, noise_se_outputs, noise_se_inputs):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    problem = BraninCurrin(negate=True).to(**tkwargs)
    train_x = duplicate(train_x, number_of_repeats)
    noisy_train_x = get_noisy_train_x(train_x, noise_se_inputs)
    train_obj_true = problem(noisy_train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * noise_se_outputs

    train_obj_true_averaged = torch.zeros([number_of_samples, 2], dtype=torch.float64)
    train_obj_averaged = torch.zeros([number_of_samples, 2], dtype=torch.float64)

    for i in range(number_of_samples):
        sum_0 = 0
        sum_1 = 0
        for j in range(1, number_of_repeats + 1):
            sum_0 += train_obj_true[i + j * number_of_samples, 0]
            sum_1 += train_obj_true[i + j * number_of_samples, 1]
        train_obj_true_averaged[i, 0] = sum_0 / number_of_repeats
        train_obj_true_averaged[i, 1] = sum_1 / number_of_repeats

        sum_0 = 0
        sum_1 = 0
        for j in range(1, number_of_repeats + 1):
            sum_0 += train_obj[i + j * number_of_samples, 0]
            sum_1 += train_obj[i + j * number_of_samples, 1]
        train_obj_averaged[i, 0] = sum_0 / number_of_repeats
        train_obj_averaged[i, 1] = sum_1 / number_of_repeats

    return train_obj_true_averaged, train_obj_averaged

def get_outputs_first_round(save_x_location, number_of_repeats, number_of_samples, noise_se_outputs, noise_se_inputs, all_outputs_location, all_true_outputs_location):
    train_x = torch.load(save_x_location)
    train_x = duplicate(train_x, number_of_repeats)
    train_obj_true, train_obj = get_outputs(train_x, number_of_samples, number_of_repeats, noise_se_outputs, noise_se_inputs)
    torch.save(train_obj, all_outputs_location)
    torch.save(train_obj_true, all_true_outputs_location)

def get_outputs_subsequent_rounds(save_x_location, number_of_repeats, number_of_samples, noise_se_outputs, noise_se_inputs, all_outputs_location, all_true_outputs_location):
    train_x = torch.load(save_x_location)
    train_x = duplicate(train_x, number_of_repeats)
    train_obj_true, train_obj = get_outputs(train_x, number_of_samples, number_of_repeats, noise_se_outputs, noise_se_inputs)

    train_obj_all = torch.load(all_outputs_location)
    train_obj_all = torch.cat([train_obj_all, train_obj])
    torch.save(train_obj_all, all_outputs_location)

    train_obj_true_all = torch.load(all_true_outputs_location)
    train_obj_true_all = torch.cat([train_obj_true_all, train_obj_true])
    torch.save(train_obj_true_all, all_true_outputs_location)

def get_data(train_x_location, train_obj_location):
    train_x = torch.load(train_x_location)
    train_obj = torch.load(train_obj_location)

    return train_x, train_obj

def initialize_model(train_x, train_obj, bounds, noise_se):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # define models for objective and constraint
    train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]
        train_yvar = torch.full_like(train_y, noise_se[i] ** 2)
        models.append(
            FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    return mll, model

def optimize_qnehvi_and_get_values_to_sample(model, train_x, sampler, batch_size, num_restarts, raw_samples, bounds, ref_point, standard_bounds):

    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # use known reference point
        X_baseline=normalize(train_x, bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # get new values to sample
    new_x = unnormalize(candidates.detach(), bounds=bounds)

    return new_x

def run_optimization(train_x_location, all_train_x_location, all_train_obj_location, batch_size, num_restarts, raw_samples, mc_samples, noise_se, problem_dimensions, bounds, ref_point, standard_bounds):
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.monotonic()
    train_x, train_obj = get_data(all_train_x_location, all_train_obj_location)
    mll, model = initialize_model(train_x, train_obj, bounds, noise_se)
    fit_gpytorch_mll(mll)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    new_x = optimize_qnehvi_and_get_values_to_sample(model, train_x, sampler, batch_size, num_restarts, raw_samples, bounds, ref_point, standard_bounds)
    train_x = torch.cat([train_x, new_x])
    torch.save(train_x, all_train_x_location)
    torch.save(new_x, train_x_location)
    t1 = time.monotonic()
    print(f"time = {t1 - t0:>4.2f}.")

def plot_outputs(initial_num_points, n_batch, batch_size, train_obj):
    pareto_front = torch.load('/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/pareto_front_outputs.pt')

    fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True, sharey=True)
    cm = plt.cm.get_cmap('viridis')

    #batch_number = torch.cat(
    #    [torch.zeros(2 * (problem.dim + 1)), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    #).numpy()
    batch_number = torch.cat(
        [torch.zeros(initial_num_points), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    ).numpy()
    axes.scatter(train_obj[:, 0].cpu().numpy(), train_obj[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)
    axes.scatter(pareto_front[:, 0].cpu().numpy(), pareto_front[:, 1].cpu().numpy(), c='grey', alpha=0.6)
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

def plot_inputs(initial_num_points, n_batch, batch_size, train_x):
    pareto_front_inputs = torch.load('/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/pareto_front_inputs.pt')

    fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True, sharey=True)
    cm = plt.cm.get_cmap('viridis')

    #batch_number = torch.cat(
    #    [torch.zeros(2 * (problem.dim + 1)), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    #).numpy()
    batch_number = torch.cat(
        [torch.zeros(initial_num_points), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    ).numpy()
    axes.scatter(train_x[:, 0].cpu().numpy(), train_x[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)
    axes.scatter(pareto_front_inputs[:, 0].cpu().numpy(), pareto_front_inputs[:, 1].cpu().numpy(), c='grey', alpha=0.6)
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

def plot(all_train_x_location, all_train_obj_location, all_train_obj_true_location, initial_num_points, batch_size, n_batch):
    train_x = torch.load(all_train_x_location)
    train_obj = torch.load(all_train_obj_location)
    train_obj_true = torch.load(all_train_obj_true_location)
    plot_outputs(initial_num_points, n_batch, batch_size, train_obj_true)
    plot_outputs(initial_num_points, n_batch, batch_size, train_obj)
    plot_inputs(initial_num_points, n_batch, batch_size, train_x)

def main():
    num_samples, number_of_repeats, number_of_cycles, bounds, noise_se_outputs, noise_se_inputs, num_restarts, raw_samples, mc_samples, problem_dimensions, ref_point, standard_bounds, save_x_location, all_x_save_location, all_true_outputs_location, all_outputs_location = inputs()
    generate_initial_inputs(bounds, num_samples, save_x_location, all_x_save_location)
    get_outputs_first_round(save_x_location, number_of_repeats, num_samples, noise_se_outputs, noise_se_inputs,
                            all_outputs_location, all_true_outputs_location)
    for i in range(number_of_cycles):
        run_optimization(save_x_location, all_x_save_location, all_outputs_location, num_samples, num_restarts,
                         raw_samples, mc_samples, noise_se_outputs, problem_dimensions, bounds, ref_point, standard_bounds)
        get_outputs_subsequent_rounds(save_x_location, number_of_repeats, num_samples, noise_se_outputs, noise_se_inputs,
                                all_outputs_location, all_true_outputs_location)
    plot(all_x_save_location, all_outputs_location, all_true_outputs_location, num_samples, num_samples,
         number_of_cycles)

main()