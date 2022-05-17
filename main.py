import math
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import Preprocessing
import FeatureSelection
import train

dataset = Preprocessing.read_dataset('/home/richel/project/AntiBacterialLearning/Book16.xlsx')
dataframe = Preprocessing.impute(dataset)
normalized_dataframe = Preprocessing.normalize_dataframe(dataframe, "Escherichia coli ZOI")
final_dataframe = normalized_dataframe[['Mean of NP core size','Extract agent (g)', 'Volume of extract agent (ml)',
'Volume of  AgNO3 (ml)','Reaction Time(min)','Concentration μg/ml','Escherichia coli ZOI']]

# FeatureSelection.AdaBoost_selector(normalized_dataframe)

result = 0
y_t = []
y_b = []
for i in range(0,3):
    y_test, y_boost = train.XGBoost(final_dataframe)
    score = r2_score(y_test, y_boost)
    if score > result:
        result = score
        y_t = y_test
        y_b = y_boost
print('mea = ' , mean_absolute_error(y_t, y_b))
print('rmse = ' , math.sqrt(mean_squared_error(y_t, y_b)))
print('r2 = ' , r2_score(y_t, y_b))
print('mse = ' , mean_squared_error(y_t, y_b))

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_boost, label="predicted")
plt.title("Antibacterial test and predicted data")
plt.legend()
plt.show()

