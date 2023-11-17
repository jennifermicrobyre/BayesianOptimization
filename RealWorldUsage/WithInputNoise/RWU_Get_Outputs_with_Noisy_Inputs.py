
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

    noise_se_outputs = torch.tensor([0.5, 0.05], **tkwargs)
    noise_se_inputs = torch.tensor([0.05, 0.05], **tkwargs)

    return first_round, inputs_location, all_true_outputs_location, all_outputs_location, noise_se_outputs, noise_se_inputs

def get_noisy_train_x(train_x, noise_se_inputs):
    noisy_train_x = train_x + torch.randn_like(train_x) * noise_se_inputs
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            if noisy_train_x[i, j] < 0 or noisy_train_x[i, j] > 1:
                noisy_train_x[i, j] = train_x[i, j]

    print(noisy_train_x)

    return noisy_train_x

def get_outputs(train_x, noise_se_outputs, noise_se_inputs):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    problem = BraninCurrin(negate=True).to(**tkwargs)
    noisy_train_x = get_noisy_train_x(train_x, noise_se_inputs)
    train_obj_true = problem(noisy_train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * noise_se_outputs

    return train_obj_true, train_obj

def main():
    first_round, inputs_location, all_true_outputs_location, all_outputs_location, noise_se_outputs, noise_se_inputs = inputs()
    train_x = torch.load(inputs_location)
    train_obj_true, train_obj = get_outputs(train_x, noise_se_outputs, noise_se_inputs)
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