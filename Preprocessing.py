#Preprocessing dataset
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import minmax_scale, normalize, scale, StandardScaler

def read_dataset(dataset_path: str):
    dataframe = pd.read_excel(dataset_path)

    #deleting reference, NP core size columns from data_frame
    dataframe.drop('References', axis=1, inplace=True)
    dataframe.drop('NP', axis=1, inplace=True)
    dataframe.drop('Nanoparticle size ', axis=1, inplace=True)

    return dataframe
    
def impute(dataframe):

    #converting categorical variable to numeric ('Method' column)
    dataframe['Method'].replace(['well', 'disk '], [0, 1], inplace=True)

    #replace * with Nan for simplicity
    dataframe.replace('*', np.nan, inplace=True)

    #filling missing values of 'Extract mass (g)' & 'Solvent volume (ml)' column
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(dataframe[['Extract mass (g)','Solvent volume (ml)']])
    dataframe[['Extract mass (g)','Solvent volume (ml)']] = imputer.transform(dataframe[['Extract mass (g)','Solvent volume (ml)']])

    #filling missing values of 'Reaction Time(min)' & 'Method' column
    time_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    time_imputer.fit(dataframe[['Reaction time(min)','Method']])
    dataframe[['Reaction time(min)','Method']] = time_imputer.transform(dataframe[['Reaction time(min)','Method']])

    return dataframe

def prepare_data(dataframe, result_name):
    features = dataframe.iloc[:,:-5]
    result = dataframe[[result_name]].squeeze()
    prepared_data = pd.concat([features, result], axis=1)
    prepared_data.dropna(subset = ["Escherichia coli ZOI"], inplace=True)
    prepared_data.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='')
    prepared_data.drop('index',axis=1, inplace=True)

    return prepared_data

def normalize_dataframe(dataframe, result_name):
    raw_data = prepare_data(dataframe, result_name)
    features = raw_data.iloc[:,:-1]
    norm_data = minmax_scale(features, feature_range=(0,1))
    normalize_features = pd.DataFrame(norm_data, index=features.index, columns=features.columns)
    result = raw_data.iloc[:,-1]
    normalize_dataframe = pd.concat([normalize_features, result], axis=1)

    return normalize_dataframe