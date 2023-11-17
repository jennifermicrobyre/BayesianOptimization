from ax import *

import numpy as np

import plot_pareto
from ax.plot.pareto_utils import compute_posterior_pareto_frontier

from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

# Factory methods for creating multi-objective optimization modesl.
from ax.modelbridge.factory import get_MOO_EHVI, get_MOO_PAREGO

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

x1 = RangeParameter(name="x1", lower=0, upper=1, parameter_type=ParameterType.FLOAT)
x2 = RangeParameter(name="x2", lower=0, upper=1, parameter_type=ParameterType.FLOAT)

search_space = SearchSpace(
    parameters=[x1, x2],
)

from botorch.test_functions.multi_objective import BraninCurrin
import torch

branin_currin = BraninCurrin(negate=True).to(
    dtype=torch.double,
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

class MetricA(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return float(branin_currin(torch.tensor(x))[0])


class MetricB(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return float(branin_currin(torch.tensor(x))[1])


metric_a = MetricA("a", ["x1", "x2"], noise_sd=0.0, lower_is_better=False)
metric_b = MetricB("b", ["x1", "x2"], noise_sd=0.0, lower_is_better=False)

mo = MultiObjective(
    objectives=[Objective(metric=metric_a), Objective(metric=metric_b)],
)

objective_thresholds = [
    ObjectiveThreshold(metric=metric, bound=val, relative=False)
    for metric, val in zip(mo.metrics, branin_currin.ref_point)
]

optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
)

# Reasonable defaults for number of quasi-random initialization points and for subsequent model-generated trials.
N_INIT = 6
N_BATCH = 25

def build_experiment():
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment

##################
# Initialize with SOBOL

def initialize_experiment(experiment):
    sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)

    for _ in range(N_INIT):
        experiment.new_trial(sobol.gen(1)).run()

    return experiment.fetch_data()

####################
# run with sobol

sobol_experiment = build_experiment()
sobol_data = initialize_experiment(sobol_experiment)

sobol_model = Models.SOBOL(
    experiment=sobol_experiment,
    data=sobol_data,
)
sobol_hv_list = []
for i in range(N_BATCH):
    generator_run = sobol_model.gen(1)
    trial = sobol_experiment.new_trial(generator_run=generator_run)
    trial.run()
    exp_df = exp_to_df(sobol_experiment)
    outcomes = np.array(exp_df[['a', 'b']], dtype=np.double)
    # Fit a GP-based model in order to calculate hypervolume.
    # We will not use this model to generate new points.
    dummy_model = get_MOO_EHVI(
        experiment=sobol_experiment,
        data=sobol_experiment.fetch_data(),
    )
    try:
        hv = observed_hypervolume(modelbridge=dummy_model)
    except:
        hv = 0
        print("Failed to compute hv")
    sobol_hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

sobol_outcomes = np.array(exp_to_df(sobol_experiment)[['a', 'b']], dtype=np.double)
print("sobol outcomes: ", sobol_outcomes)

frontier = compute_posterior_pareto_frontier(
    experiment=sobol_experiment,
    data=sobol_experiment.fetch_data(),
    primary_objective=metric_b,
    secondary_objective=metric_a,
    absolute_metrics=["a", "b"],
    num_points=20,
)

fig = plot_pareto.plot_pareto_frontier(frontier, CI_level=0.90)
fig.show()

##############
# qNEHVI
ehvi_experiment = build_experiment()
ehvi_data = initialize_experiment(ehvi_experiment)

ehvi_hv_list = []
ehvi_model = None
for i in range(N_BATCH):
    ehvi_model = get_MOO_EHVI(
        experiment=ehvi_experiment,
        data=ehvi_data,
    )
    print("evhi model: ", ehvi_model)
    generator_run = ehvi_model.gen(1)
    print('generator run: ', generator_run)
    trial = ehvi_experiment.new_trial(generator_run=generator_run)
    print("trial: ", trial)
    trial.run()
    print("trial.run(): ", trial)
    print("trial fetch: ", trial.fetch_data())
    ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])
    print("evhi_data: ", ehvi_data)

    exp_df = exp_to_df(ehvi_experiment)
    print("exp_df: ", exp_df)
    outcomes = np.array(exp_df[['a', 'b']], dtype=np.double)
    try:
        hv = observed_hypervolume(modelbridge=ehvi_model)
    except:
        hv = 0
        print("Failed to compute hv")
    ehvi_hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

exp_df.to_csv('/Users/jennifer/Desktop/exp_df.csv')
print(float(branin_currin(torch.tensor([0.10299069464854200, 0.885408387784519]))[0]))
print(float(branin_currin(torch.tensor([0.10299069464854200, 0.885408387784519]))[1]))

ehvi_outcomes = np.array(exp_to_df(ehvi_experiment)[['a', 'b']], dtype=np.double)
print("EVHI outcomes")
print(ehvi_outcomes)

frontier = compute_posterior_pareto_frontier(
    experiment=ehvi_experiment,
    data=ehvi_experiment.fetch_data(),
    primary_objective=metric_b,
    secondary_objective=metric_a,
    absolute_metrics=["a", "b"],
    num_points=20,
)

fig = plot_pareto.plot_pareto_frontier(frontier, CI_level=0.90)
fig.show()


##########
# qNParEGO

parego_experiment = build_experiment()
parego_data = initialize_experiment(parego_experiment)

parego_hv_list = []
parego_model = None
for i in range(N_BATCH):
    parego_model = get_MOO_PAREGO(
        experiment=parego_experiment,
        data=parego_data,
    )
    generator_run = parego_model.gen(1)
    trial = parego_experiment.new_trial(generator_run=generator_run)
    trial.run()
    parego_data = Data.from_multiple_data([parego_data, trial.fetch_data()])

    exp_df = exp_to_df(parego_experiment)
    outcomes = np.array(exp_df[['a', 'b']], dtype=np.double)
    try:
        hv = observed_hypervolume(modelbridge=parego_model)
    except:
        hv = 0
        print("Failed to compute hv")
    parego_hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

parego_outcomes = np.array(exp_to_df(parego_experiment)[['a', 'b']], dtype=np.double)

frontier = compute_posterior_pareto_frontier(
    experiment=parego_experiment,
    data=parego_experiment.fetch_data(),
    primary_objective=metric_b,
    secondary_objective=metric_a,
    absolute_metrics=["a", "b"],
    num_points=20,
)

fig = plot_pareto.plot_pareto_frontier(frontier, CI_level=0.90)
fig.show()


####################
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

fig, axes = plt.subplots(1, 3, figsize=(20,6))
algos = ["Sobol", "qNParEGO", "qNEHVI"]
outcomes_list = [sobol_outcomes, parego_outcomes, ehvi_outcomes]
cm = plt.cm.get_cmap('viridis')
BATCH_SIZE = 1

n_results = N_BATCH*BATCH_SIZE + N_INIT
batch_number = torch.cat([torch.zeros(N_INIT), torch.arange(1, N_BATCH+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]).numpy()
for i, train_obj in enumerate(outcomes_list):
    x = i
    sc = axes[x].scatter(train_obj[:n_results, 0], train_obj[:n_results,1], c=batch_number[:n_results], alpha=0.8)
    axes[x].set_title(algos[i])
    axes[x].set_xlabel("Objective 1")
    axes[x].set_xlim(-150, 5)
    axes[x].set_ylim(-15, 0)
axes[0].set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm =  ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")
plt.show()