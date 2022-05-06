from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression

def XGBoost(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = XGBRegressor()
    model.fit(x_train, y_train)
    y_boost = model.predict(x_test)

    return y_test, y_boost
    
def DecisionTree(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = tree.DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_decision = model.predict(x_test)
    
    return y_test, y_decision


def RandomForest(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(x_train, y_train.values.ravel())
    y_random = model.predict(x_test)
    
    return y_test, y_random

def Adaboost(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = AdaBoostRegressor(random_state=50, n_estimators=100)
    model.fit(x_train, y_train)
    y_adaboost = model.predict(x_test)

    return y_test, y_adaboost


