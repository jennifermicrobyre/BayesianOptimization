
import pandas as pd
import yaml

def inputs():
    file_to_open = 'BL-7QGEY1I0RM_layout.yaml'
    file_open_loc = '/Users/jennifer/Documents/Bayesian_Optimization/Glycerol_and_Bicarbonate/'

    file_to_save = 'BL-7QGEY1I0RM_layout.csv'
    file_save_loc = '/Users/jennifer/Documents/Bayesian_Optimization/Glycerol_and_Bicarbonate/'

    return file_to_open, file_open_loc, file_to_save, file_save_loc

def open_yaml(file_open_loc, filename):

    with open(file_open_loc + filename) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)

    return data


def get_layout(data):
    #wells = list(data["Wells"].keys())
    well_nums = list(data['Plates']['BL-7QGEY1I0RM.plate-01']['Wells'].keys())
    inoculant = []
    condition = []
    for i in range(len(well_nums)):
        inoculant.append(data['Plates']['BL-7QGEY1I0RM.plate-01']['Wells'][well_nums[i]]['inoculant'])
        condition.append(data['Plates']['BL-7QGEY1I0RM.plate-01']['Wells'][well_nums[i]]['variant'])
    df = pd.DataFrame({'wellnum': well_nums, 'Inoculant': inoculant, 'Condition': condition})

    return df

def main():
    file_to_open, file_open_loc, file_to_save, file_save_loc = inputs()
    data = open_yaml(file_open_loc, file_to_open)
    df = get_layout(data)
    df.to_csv(file_save_loc + file_to_save, index=False)



main()