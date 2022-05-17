from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler

    
def DecisionTree(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = DecisionTreeRegressor()
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

def XGBoost(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = XGBRegressor()
    model.fit(x_train, y_train)
    y_boost = model.predict(x_test)

    return y_test, y_boost

def AdaBoost(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = AdaBoostRegressor(random_state=50, n_estimators=80)
    model.fit(x_train, y_train)
    y_adaboost = model.predict(x_test)
    
    return y_test, y_adaboost
