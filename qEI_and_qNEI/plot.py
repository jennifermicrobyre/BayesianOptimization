import torch
from matplotlib import pyplot as plt
import numpy as np

def inputs():
    train_obj_csv_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj.csv'
    train_obj_all_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj_all.pt'
    train_x_all_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x_all.pt'

    return train_obj_csv_location, train_obj_all_location, train_x_all_location

def load_obj_make_tensor(train_obj_location, train_obj_all_location):
    train_obj = np.loadtxt(train_obj_location, delimiter=",")
    train_obj = torch.from_numpy(train_obj)
    if torch.cuda.is_available():
        train_obj = train_obj.to('cuda')

    train_obj = torch.reshape(train_obj, (len(train_obj), 1))

    train_obj_all = torch.load(train_obj_all_location)
    train_obj_all = torch.cat([train_obj_all, train_obj])

    return train_obj_all

def plot_scatter(train_x_all, train_obj_all):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.scatter(train_x_all[:108, 0], train_x_all[:108, 1], c=train_obj_all, cmap='viridis_r')
    #ax.invert_yaxis()

    plt.show()

def main():
    train_obj_csv_location, train_obj_all_location, train_x_all_location = inputs()
    train_x_all = torch.load(train_x_all_location)
    train_obj_all = torch.load(train_obj_all_location)
    #train_obj_all = load_obj_make_tensor(train_obj_csv_location, train_obj_all_location)
    plot_scatter(train_x_all, train_obj_all)

main()