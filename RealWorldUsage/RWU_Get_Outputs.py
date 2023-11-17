
import torch
from botorch.test_functions.multi_objective import BraninCurrin

def inputs():
    first_round = 0
    inputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x.pt'
    all_true_outputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all_true.pt'
    all_outputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all.pt'

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    noise_se = torch.tensor([5, 0.5], **tkwargs)

    return first_round, inputs_location, all_true_outputs_location, all_outputs_location, noise_se

def get_outputs(train_x, noise_se):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    problem = BraninCurrin(negate=True).to(**tkwargs)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * noise_se

    return train_obj_true, train_obj

def main():
    first_round, inputs_location, all_true_outputs_location, all_outputs_location, noise_se = inputs()
    train_x = torch.load(inputs_location)
    train_obj_true, train_obj = get_outputs(train_x, noise_se)
    if first_round == 0:
        train_obj_all = torch.load(all_outputs_location)
        train_obj_all = torch.cat([train_obj_all, train_obj])
        torch.save(train_obj_all, all_outputs_location)

        train_obj_true_all = torch.load(all_true_outputs_location)
        train_obj_true_all = torch.cat([train_obj_true_all, train_obj_true])
        torch.save(train_obj_true_all, all_true_outputs_location)
    else:
        torch.save(train_obj, all_outputs_location)
        torch.save(train_obj_true, all_true_outputs_location)

main()