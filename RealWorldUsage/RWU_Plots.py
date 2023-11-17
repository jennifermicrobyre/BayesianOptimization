
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

def inputs():
    all_train_x_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x_all.pt'
    all_train_obj_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all.pt'
    all_train_obj_true_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all_true.pt'

    initial_num_points = 11
    batch_size = 11
    n_batch = 3

    return all_train_x_location, all_train_obj_location, all_train_obj_true_location, initial_num_points, batch_size, n_batch

def plot_outputs(initial_num_points, n_batch, batch_size, train_obj):
    fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True, sharey=True)
    cm = plt.cm.get_cmap('viridis')

    #batch_number = torch.cat(
    #    [torch.zeros(2 * (problem.dim + 1)), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    #).numpy()
    batch_number = torch.cat(
        [torch.zeros(initial_num_points), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    ).numpy()
    axes.scatter(train_obj[:, 0].cpu().numpy(), train_obj[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)
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
    fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True, sharey=True)
    cm = plt.cm.get_cmap('viridis')

    #batch_number = torch.cat(
    #    [torch.zeros(2 * (problem.dim + 1)), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    #).numpy()
    batch_number = torch.cat(
        [torch.zeros(initial_num_points), torch.arange(1, n_batch + 1).repeat(batch_size, 1).t().reshape(-1)]
    ).numpy()
    axes.scatter(train_x[:, 0].cpu().numpy(), train_x[:, 1].cpu().numpy(), c=batch_number, alpha=0.8)
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

def main():
    all_train_x_location, all_train_obj_location, all_train_obj_true_location, initial_num_points, batch_size, n_batch = inputs()
    train_x = torch.load(all_train_x_location)
    train_obj = torch.load(all_train_obj_location)
    train_obj_true = torch.load(all_train_obj_true_location)
    plot_outputs(initial_num_points, n_batch, batch_size, train_obj_true)
    plot_outputs(initial_num_points, n_batch, batch_size, train_obj)
    plot_inputs(initial_num_points, n_batch, batch_size, train_x)

main()

