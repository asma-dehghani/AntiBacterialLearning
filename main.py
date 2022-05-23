import math
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import Preprocessing
import FeatureSelection
import train
import pandas as pd
import Charts

dataset = Preprocessing.read_dataset('/home/richel/project/AntiBacterialLearning/Book16.xlsx')
dataframe = Preprocessing.impute(dataset)
normalized_dataframe = Preprocessing.normalize_dataframe(dataframe, "Escherichia coli ZOI")
final_dataframe = normalized_dataframe[['Mean of NP core size', 'AgNO3 volume  (ml)', 'Reaction temperature (0C)', 
'Reaction time(min)', 'Method', 'Nanoparticles concentration (μg/ml)', 'Escherichia coli ZOI']]

x_train, x_test, y_train, y_test, y_predict = train.DecisionTree(final_dataframe)

print('mea = ' , mean_absolute_error(y_test, y_predict))
print('mse = ' , mean_squared_error(y_test, y_predict))
print('rmse = ' , math.sqrt(mean_squared_error(y_test, y_predict)))
print('r2 = ' , r2_score(y_test, y_predict))

Charts.prediction_result(y_test, y_predict)
