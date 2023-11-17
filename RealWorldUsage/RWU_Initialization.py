import torch
import pandas as pd
from botorch.utils.sampling import draw_sobol_samples

def inputs():
    num_samples = 12
    # use this for glyercol and Bicarbonate bounds = [[50.0, 30.0], [300.0, 250.0]]
    dimensions = 2
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    bounds = torch.tensor([[-5] * dimensions, [5] * dimensions], device=device, dtype=dtype)

    save_location = '/Users/jennifer/Documents/Bayesian_Optimization/Electroporation/train_x.pt'
    all_save_location = '/Users/jennifer/Documents/Bayesian_Optimization/Electroporation/train_x_all.pt'
    csv_save_location = '/Users/jennifer/Documents/Bayesian_Optimization/Electroporation/starting_conditions.csv'

    #############
    print(f"Device tensor is stored on: {bounds.device}")

    return num_samples, bounds, save_location, all_save_location, csv_save_location

def convert_to_csv(train_x):
    train_x_np = train_x.numpy()
    train_x_df = pd.DataFrame(train_x_np)

    return train_x_df

def main():
    num_samples, bounds, save_location, all_save_location, csv_save_location = inputs()
    train_x = draw_sobol_samples(bounds=bounds, n=num_samples, q=1).squeeze(1)
    train_x_df = convert_to_csv(train_x)
    torch.save(train_x, save_location)
    torch.save(train_x, all_save_location)
    train_x_df.to_csv(csv_save_location)


main()