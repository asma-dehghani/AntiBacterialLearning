#Selecting important features
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.feature_selection import RFE, SelectFromModel 
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import pandas as pd

def get_feature_name(features):
    features.columns = ['core.size.avg', 'agent', 'solvent','agent.vol', 'Ag.vol', 'Ag.concentration', 'react.temp',  
            'react.time','diameters' ,'incubate.temp', 'incubate.time', 'Cell density of the inoculum', 'method', 'np.concentration']
    names = list()
    for column in features:
        names.append(column)

    return names

def Decisiontree_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    decison_tree = DecisionTreeRegressor()
    model = RFE(estimator=decison_tree, n_features_to_select=14)
    model.fit(features, result)
    importance = model.estimator_.feature_importances_
    feature_name = get_feature_name(features)
    importance_df=pd.DataFrame({'Name':feature_name, 'Importance':importance})
    importance_df.sort_values(by=['Importance'], ascending=False, inplace=True)
    sns.barplot(x='Importance', y='Name', data=importance_df)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Name')
    plt.show()

def RandomForest_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    random_forest = RandomForestRegressor()
    model = RFE(estimator=random_forest, n_features_to_select=14)
    model.fit(features, result)
    importance = model.estimator_.feature_importances_
    feature_name = get_feature_name(features)
    importance_df=pd.DataFrame({'Name':feature_name, 'Importance':importance})
    sorted_df=importance_df.sort_values(by=['Importance'], ascending=False)
    sns.barplot(x='Importance', y='Name', data=sorted_df)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Name')
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
    