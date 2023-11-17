import torch
import numpy as np

def inputs():
    save_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x.pt'
    all_save_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x_all.pt'
    csv_open_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/starting_conditions.csv'

    return save_location, all_save_location, csv_open_location

def main():
    save_location, all_save_location, csv_open_location = inputs()
    train_x = np.loadtxt(csv_open_location, delimiter=",")
    train_x = torch.from_numpy(train_x)
    if torch.cuda.is_available():
        train_x = train_x.to('cuda')
    torch.save(train_x, save_location)
    torch.save(train_x, all_save_location)

main()