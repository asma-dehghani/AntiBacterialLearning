import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

columns = ["Algorithms", "MAE", "RMSE", "R2"]
test_data = pd.DataFrame.from_records([("DecisionTree", 2.38, 3.0, 0.73),\
                                       ("RandomForest", 2.74, 3.53, 0.68),\
                                       ("XGBoost", 1.89, 2.46, 0.78),\
                                       ("AdaBoost", 2.55, 3.12, 0.71)],\
                                       columns=columns)
test_data_r2 = test_data["R2"]
test_data_mae = test_data["MAE"]
test_data_rmse = test_data["RMSE"]
#test_data_other = test_data.iloc[:,:-1]


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
x = test_data["Algorithms"]

X_axis = np.arange(len(x))
width = 0.2
ax1.bar(X_axis - width, test_data_mae, 0.2, color = ['g'])
ax1.bar(X_axis, test_data_rmse, 0.2, color = 'blue')
ax2.bar(X_axis + width, test_data_r2, 0.2, color = 'yellow')
plt.xticks(X_axis, x)
pos = np.arange(len(x))
plt.xticks(pos, x)
ax1.set_ylabel('RMSE, MAE')
ax2.set_ylabel('R ^ 2')
ax1.set_ylim(0,10)
ax2.set_ylim(0,1)

plt.show()

