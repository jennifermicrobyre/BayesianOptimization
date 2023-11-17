import torch
from botorch.test_functions import StyblinskiTang
import pandas as pd

location_x = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_x.pt'
location_obj = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj.pt'
location_obj_all = '/Users/jennifer/Documents/Bayesian_Optimization/Single_Objective_Testing/train_obj_all.pt'

new_x = torch.load(location_x)

synthetic_test_function = StyblinskiTang(negate=True)
new_obj = synthetic_test_function(new_x).unsqueeze(-1)

new_obj_np = new_obj.numpy()
new_obj_df = pd.DataFrame(new_obj_np)
new_obj_df.to_csv(location_obj[:-2] + 'csv', header=False, index=False)

torch.save(new_obj, location_obj)

train_obj_all = torch.load(location_obj_all)
train_obj_all = torch.cat([train_obj_all, new_obj])
torch.save(train_obj_all, location_obj_all)
