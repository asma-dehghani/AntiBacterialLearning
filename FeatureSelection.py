#Selecting important features
from operator import mod
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.tree import DecisionTreeRegressor
import numpy as np

def get_feature_name(features):
    features.columns = ['core.size.avg', 'agent', 'solvent','agent.vol', 'Ag.vol', 'Ag.concentration', 'react.temp',  
            'react.time','diameters' ,'incubate.temp', 'incubate.time', 'cell.density', 'method', 'np.concentration']
    names = list()
    for column in features:
        names.append(column)

    return names

def XGBoost_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    model = XGBRegressor()
    model.fit(features, result)
    # print(model.feature_importances_)
    # plot_importance(model)
    # plt.show()
    importance = np.abs(model.feature_importances_)
    feature_name = get_feature_name(features)
    plt.barh(feature_name, importance)
    plt.title("XGBoost Feature Ranking")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
    
    # plot_importance(model)
    # return plt.show()

def rfe_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    model = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=6)
    model.fit(features, result)
    importance = model.ranking_
    feature_name = get_feature_name(features)
    plt.barh(feature_name, importance)
    print(model.support_)
    plt.title("RFE Feature Ranking")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

def ada_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    estimator = AdaBoostRegressor(random_state=50, n_estimators=100)
    selector = SelectFromModel(estimator)
    selector = selector.fit(features, result)
    print(selector.get_support())

