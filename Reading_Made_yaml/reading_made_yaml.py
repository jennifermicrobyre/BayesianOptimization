import pandas as pd
import yaml

# !!! Currently this code assumes that all of the concentrations have the same unit

def inputs():
    file_to_open = 'BL-7F9ETB1VW7_made.yaml'
    file_open_loc = '/Users/jennifer/Documents/Bayesian_Optimization/Glycerol_and_Bicarbonate/'

    file_to_save = 'BL-7F9ETB1VW7_made.csv'
    file_save_loc = '/Users/jennifer/Documents/Bayesian_Optimization/Glycerol_and_Bicarbonate/'

    return file_to_open, file_open_loc, file_to_save, file_save_loc

def open_yaml(file_open_loc, filename):

    with open(file_open_loc + filename) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)

    return data

def get_concentrations(data):
    conditions = list(data.keys())
    df = pd.DataFrame({'Conditions': conditions})
    number_of_molecules = len(data[conditions[0]]['variation'])
    for i in range(number_of_molecules):
        array = []
        for j in range(len(conditions)):
            amount = (data[conditions[j]]['variation'][i]['amount'])
            array.append(float(amount[:-3]))
        df.insert(i + 1, data[conditions[0]]['variation'][i]['name'], array)
    
    return df

def main():
    file_to_open, file_open_loc, file_to_save, file_save_loc = inputs()
    data = open_yaml(file_open_loc, file_to_open)
    df = get_concentrations(data)
    df.to_csv(file_save_loc + file_to_save, index=False)

main()
