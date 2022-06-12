#Selecting important features
from operator import index
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.feature_selection import RFE, SelectFromModel 
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import pandas as pd
import Charts

def get_feature_name(features):
    features.columns = ['Average Nanoparticle Size', 'Extract Mass', 'Solvent Volume','Extract Volume ', 'AgNO3 Volume ', 'AgNO3 Concentration', 'Reaction Temperature',  
            'Reaction Time','Diameters of Disks and Wells' ,'Incubation Temperature', 'Incubation Time', 'Bacteria Concentration', 'Method', 'Nanoparticles Concentration']
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
    print(importance_df)
    Charts.feature_importnace(importance_df)

def RandomForest_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    random_forest = RandomForestRegressor()
    model = RFE(estimator=random_forest, n_features_to_select=14)
    model.fit(features, result)
    importance = model.estimator_.feature_importances_
    feature_name = get_feature_name(features)
    importance_df=pd.DataFrame({'Name':feature_name, 'Importance':importance})
    importance_df.sort_values(by=['Importance'], ascending=False, inplace=True)
    Charts.feature_importnace(importance_df)

def XGBoost_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    model = XGBRegressor()
    model.fit(features, result)
    feature_name = get_feature_name(features)
    importance = model.get_booster().get_score()
    importance_df = pd.DataFrame.from_dict({'Name': feature_name, 'Importance': importance.values()})
    importance_df.sort_values(by=['Importance'], ascending=False, inplace=True)
    Charts.feature_importnace(importance_df)

def AdaBoost_selector(dataframe):
    features = dataframe.iloc[:,:-1]
    result = dataframe.iloc[:,-1]
    estimator = AdaBoostRegressor()
    model = SelectFromModel(estimator)
    model = model.fit(features, result)
    feature_name = get_feature_name(features)
    importance = model.estimator_.feature_importances_
    importance_df=pd.DataFrame({'Name':feature_name, 'Importance':importance})
    print(importance_df)
    importance_df.sort_values(by=['Importance'], ascending=False, inplace=True)
    Charts.feature_importnace(importance_df)