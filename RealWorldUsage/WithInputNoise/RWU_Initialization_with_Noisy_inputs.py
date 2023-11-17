import torch
import pandas as pd
from botorch.utils.sampling import draw_sobol_samples

def inputs():
    num_samples = 16
    # use this for glyercol and Bicarbonate bounds = [[50.0, 30.0], [300.0, 250.0]]
    bounds = torch.tensor([[0, 0], [1, 1]], dtype=torch.float64)

    save_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x.pt'
    all_save_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x_all.pt'
    csv_save_location = '/Users/jennifer/Documents/Bayesian_Optimization/Glycerol_and_Bicarbonate/starting_conditions.csv'

    #############
    #bounds = torch.tensor(bounds, dtype=torch.float64)
    if torch.cuda.is_available():
        bounds = bounds.to('cuda')
    print(f"Device tensor is stored on: {bounds.device}")

    return num_samples, bounds, save_location, all_save_location, csv_save_location

def convert_to_csv(train_x):
    train_x_np = train_x.numpy()
    train_x_df = pd.DataFrame(train_x_np)

    return train_x_df

def main():
    num_samples, bounds, save_location, all_save_location, csv_save_location = inputs()
    train_x = draw_sobol_samples(bounds=bounds, n=num_samples, q=1).squeeze(1)
    duplicated_train_x = torch.cat((train_x, train_x, train_x), 0)
    #train_x_df = convert_to_csv(train_x)
    torch.save(duplicated_train_x, save_location)
    torch.save(duplicated_train_x, all_save_location)
    #train_x_df.to_csv(csv_save_location)


main()