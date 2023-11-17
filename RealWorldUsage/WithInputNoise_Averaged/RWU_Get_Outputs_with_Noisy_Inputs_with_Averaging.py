
import torch
from botorch.test_functions.multi_objective import BraninCurrin

def inputs():
    first_round = 0
    number_of_samples = 11
    number_of_repeats = 8
    inputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_x.pt'
    all_true_outputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all_true.pt'
    all_outputs_location = '/Users/jennifer/Documents/Bayesian_Optimization/MOO_Testing/train_obj_all.pt'

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    noise_se_outputs = torch.tensor([5, 0.5], **tkwargs)
    noise_se_inputs = torch.tensor([0.05, 0.05], **tkwargs)

    return first_round, number_of_samples, number_of_repeats, inputs_location, all_true_outputs_location, all_outputs_location, noise_se_outputs, noise_se_inputs

def duplicate(train_x, number_of_repeats):
    duplicated_train_x = torch.cat((train_x, train_x), 0)
    for i in range(number_of_repeats - 2):
        duplicated_train_x = torch.cat((duplicated_train_x, train_x), 0)

    return duplicated_train_x

def get_noisy_train_x(train_x, noise_se_inputs):
    noisy_train_x = train_x + torch.randn_like(train_x) * noise_se_inputs
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            if noisy_train_x[i, j] < 0 or noisy_train_x[i, j] > 1:
                noisy_train_x[i, j] = train_x[i, j]

    return noisy_train_x

def get_outputs(train_x, number_of_samples, number_of_repeats, noise_se_outputs, noise_se_inputs):
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    problem = BraninCurrin(negate=True).to(**tkwargs)
    train_x = duplicate(train_x, number_of_repeats)
    noisy_train_x = get_noisy_train_x(train_x, noise_se_inputs)
    train_obj_true = problem(noisy_train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * noise_se_outputs

    train_obj_true_averaged = torch.zeros([number_of_samples, 2], dtype=torch.float64)
    train_obj_averaged = torch.zeros([number_of_samples, 2], dtype=torch.float64)

    for i in range(number_of_samples):
        sum_0 = 0
        sum_1 = 0
        for j in range(1, number_of_repeats + 1):
            sum_0 += train_obj_true[i + j * number_of_samples, 0]
            sum_1 += train_obj_true[i + j * number_of_samples, 1]
        train_obj_true_averaged[i, 0] = sum_0 / number_of_repeats
        train_obj_true_averaged[i, 1] = sum_1 / number_of_repeats

        sum_0 = 0
        sum_1 = 0
        for j in range(1, number_of_repeats + 1):
            sum_0 += train_obj[i + j * number_of_samples, 0]
            sum_1 += train_obj[i + j * number_of_samples, 1]
        train_obj_averaged[i, 0] = sum_0 / number_of_repeats
        train_obj_averaged[i, 1] = sum_1 / number_of_repeats

    return train_obj_true_averaged, train_obj_averaged

def main():
    first_round, number_of_samples, number_of_repeats, inputs_location, all_true_outputs_location, all_outputs_location, noise_se_outputs, noise_se_inputs = inputs()
    train_x = torch.load(inputs_location)
    train_x = duplicate(train_x, number_of_repeats)
    train_obj_true, train_obj = get_outputs(train_x, number_of_samples, number_of_repeats, noise_se_outputs, noise_se_inputs)
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