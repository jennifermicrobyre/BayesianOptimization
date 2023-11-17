
import torch
from botorch.test_functions import StyblinskiTang
import pandas as pd

location_x = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x.pt'
location_obj = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj.csv'

new_x = torch.load(location_x)

synthetic_test_function = StyblinskiTang(negate=True)
new_obj = synthetic_test_function(new_x).unsqueeze(-1)

new_obj_np = new_obj.numpy()
new_obj_df = pd.DataFrame(new_obj_np)
new_obj_df.to_csv(location_obj, header=False, index=False)

