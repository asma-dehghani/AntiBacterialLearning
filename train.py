from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

    
def DecisionTree(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    
    return x_train, x_test, y_train, y_test, y_predict


def RandomForest(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(x_train, y_train.values.ravel())
    y_predict = model.predict(x_test)
    
    return x_train, x_test, y_train, y_test, y_predict

def XGBoost(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = XGBRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    return x_train, x_test, y_train, y_test, y_predict

def AdaBoost(dataframe):
    x = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    model = AdaBoostRegressor(n_estimators=25, random_state=42)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    
    return x_train, x_test, y_train, y_test, y_predict
