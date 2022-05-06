import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import Preprocessing
import FeatureSelection
import train

dataset = Preprocessing.read_dataset('/home/richel/Downloads/project/Book16.xlsx')
dataframe = Preprocessing.impute(dataset)
normalized_dataframe = Preprocessing.normalize_dataframe(dataframe, "Escherichia coli ZOI")
final_dataframe = normalized_dataframe[['Mean of NP core size','Solvent (ml)','Reaction Temperature (0C)','Reaction Time(min)',
'Reaction Time(min)','Concentration μg/ml', 'Method','Escherichia coli ZOI']]

selection = FeatureSelection.ada_selector(normalized_dataframe)
# y_test, y_boost = train.Adaboost(final_dataframe)
# print('mea = ' , mean_absolute_error(y_test, y_boost))
# print('rmse = ' , math.sqrt(mean_squared_error(y_test, y_boost)))
# print('r2 = ' , r2_score(y_test, y_boost))
# print('mse = ' , mean_squared_error(y_test, y_boost))

# for i in range(0, len(y_test)):
#     print(y_test.values[i],'\t', y_boost[i])

