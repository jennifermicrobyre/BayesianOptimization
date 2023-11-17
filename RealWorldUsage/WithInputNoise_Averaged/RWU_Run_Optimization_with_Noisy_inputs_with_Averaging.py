
import torch
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

def inputs():
    train_x_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x.pt'

    all_train_x_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x_all.pt'
    all_train_obj_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all.pt'

    batch_size = 11
    num_restarts = 10
    raw_samples = 512
    mc_samples = 128

    noise_se = [5, 0.5]  # noise of outputs

    problem_dimensions = 2
    bounds = [[0.0, 0.0], [1.0, 1.0]]  # used to normalize & unnormalize data
    ref_point = [-18.0,  -6.0]  # outputs  - From code: ref_point (Union[List[float], Tensor]) – A list or tensor with m elements
                                # representing the reference point (in the outcome space) w.r.t. to which compute the hypervolume.
                                # This is a reference point for the objective values (i.e. after applying objective to the samples)
                                # Also from code: ...specifying a reference point, which is the lower bound on the objectives used
                                # for computing hypervolume. ... In this tutorial, we assume the reference point is known. In practice
                                # the reference point can be set 1) using domain knowledge to be slightly worse that the lower bound
                                # of the objective values, where the lower bound is the minimum acceptable value of interest for each
                                # oobjective, or 2) using a dynamic reference point selection strategy
                                # From paper: we selected the reference point based on the component-wise noiseless nadir point

    standard_bounds = torch.zeros(2, problem_dimensions, dtype=torch.float64)
    standard_bounds[1] = 1   # A 2 x d tensor of lower and upper bounds for each column of X (if inequality_constraints is provided,
                             # these bounds can be -inf and +inf, respectively)

    ################
    noise_se = torch.tensor(noise_se, dtype=torch.float64)
    bounds = torch.tensor(bounds, dtype=torch.float64)
    ref_point = torch.tensor(ref_point, dtype=torch.float64)
    if torch.cuda.is_available():
        noise_se = noise_se.to('cuda')
        bounds = bounds.to('cuda')
        ref_point = ref_point.to('cuda')
        standard_bounds = standard_bounds.to('cuda')
    print(f"Device tensor is stored on: {bounds.device}")

    return train_x_location, all_train_x_location, all_train_obj_location, batch_size, num_restarts, raw_samples, mc_samples, noise_se, problem_dimensions, bounds, ref_point, standard_bounds

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

def main():
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.monotonic()
    train_x_location, all_train_x_location, all_train_obj_location, batch_size, num_restarts, raw_samples, mc_samples, noise_se, problem_dimensions, bounds, ref_point, standard_bounds = inputs()
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

main()