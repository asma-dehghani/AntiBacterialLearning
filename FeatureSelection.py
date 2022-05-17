#Selecting important features
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.feature_selection import RFE, SelectFromModel 
from sklearn.tree import DecisionTreeRegressor

def get_feature_name(features):
    features.columns = ['core.size.avg', 'agent', 'solvent','agent.vol', 'Ag.vol', 'Ag.concentration', 'react.temp',  
            'react.time','diameters' ,'incubate.temp', 'incubate.time', 'cell.density', 'method', 'np.concentration']
    names = list()
    for column in features:
        names.append(column)

    return names

def Decisiontree_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    model = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=6)
    model.fit(features, result)
    importance = model.ranking_
    feature_name = get_feature_name(features)
    plt.barh(feature_name, importance)
    print(model.support_)
    plt.title("DecisionTree Feature Ranking")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

def RandomForest_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    model = RFE(estimator=RandomForestRegressor(), n_features_to_select=6)
    model.fit(features, result)
    importance = model.ranking_
    feature_name = get_feature_name(features)
    plt.barh(feature_name, importance)
    print(model.support_)
    plt.title("RandomForest Feature Ranking")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

def XGBoost_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    model = XGBRegressor()
    model.fit(features, result)
    plot_importance(model)
    plt.show()

def AdaBoost_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    estimator = AdaBoostRegressor()
    model = SelectFromModel(estimator)
    model = model.fit(features, result)
    feature_name = get_feature_name(features)
    importance = model.get_support()
    plt.barh(feature_name, importance)
    plt.title("AdaBoost Feature Importance")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
    