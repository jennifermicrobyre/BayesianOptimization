import torch
import pandas as pd
from botorch.utils.sampling import draw_sobol_samples

def inputs():
    num_samples = 12

    dimensions = 2
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    bounds = torch.tensor([[0.2, 50], [3, 1000]], device=device, dtype=dtype)

    csv_save_location = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/starting_conditions.csv'

    return num_samples, bounds, csv_save_location

def convert_to_csv(train_x):
    train_x_np = train_x.numpy()
    train_x_df = pd.DataFrame(train_x_np)

    return train_x_df

def main():
    num_samples, bounds, csv_save_location = inputs()
    train_x = draw_sobol_samples(bounds=bounds, n=num_samples, q=1).squeeze(1)
    train_x_df = convert_to_csv(train_x)
    train_x_df.to_csv(csv_save_location, header=False, index=False)

main()