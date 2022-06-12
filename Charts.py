import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import FeatureSelection
import Preprocessing
import seaborn as sns
import csv

def feature_importnace(selected_features_df):
    ax = sns.barplot(x='Importance', y='Name', data=selected_features_df)
    ax.set_xlim(0,0.35)
    ax.set_xlabel("Feature Importance Score", fontsize = 16)
    ax.set_ylabel('Feature Name', fontsize = 16)
    ax.tick_params(labelsize=13)
    plt.show()

def prediction_result(y_test, y_result):
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label='Original')
    plt.plot(x_ax, y_result, label='Predicted')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.show()

def bar3D():
    with open('bar3d.csv', 'r') as stats:
        reader = csv.DictReader(stats)
        Bacteria = []
        cig1 = []
        cig2 = []
        cig3 = []
        cig4 = []
        for item in reader:
            Bacteria.append(item['Bacteria'])
            cig1.append(float(item['Escherichia Coli']))
            cig2.append(float(item['Pseudomonas Aeruginosa']))
            cig3.append(float(item['Staphylococcus Aureus']))
            cig4.append(float(item['Klebsiella Pneumonia']))


    fig = plt.figure('Feature Selection Based on RFE', figsize=(18,18))
    ax = plt.subplot(111, projection='3d')

    # x axis
    x1 = np.zeros(len(Bacteria))
    x2 = np.repeat(1, len(Bacteria))
    x3 = np.repeat(2, len(Bacteria))
    x4 = np.repeat(3,len(Bacteria))
    dx = np.repeat(0.5, len(Bacteria))

    # y oordinate
    y1 = np.arange(len(Bacteria))
    y2 = y1
    y3 = y1
    y4 = y1
    dy = 0.8

    # z coordinate
    z1 = x1
    z2 = x1
    z3 = x1
    z4 = x1
    dz1 = cig1
    dz2 = cig2
    dz3 = cig3
    dz4 = cig4

    # bar projection
    ax.bar3d(x1, y1, z1, dx, dy, dz1, color = '#003f5c', alpha = 0.7)
    ax.bar3d(x2, y2, z2, dx, dy, dz2, color = '#ffa600', alpha = 0.7)
    ax.bar3d(x3, y3, z3, dx, dy, dz3, color = 'red', alpha = 0.7)  
    ax.bar3d(x4, y4, z4, dx, dy, dz4, color = 'green', alpha = 0.7) 

    # axis naming
    ax.set_zlabel('Importance')

    # xticks rename
    plt.xticks([0.25, 1.25, 2.25, 3.25], ["Escherichia Coli", "Pseudomonas Aeruginosa", 
    "Staphylococcus Aureus", "Klebsiella Pneumonia"], rotation = 70, size = 10)

    # yticks rename
    plt.yticks(y1, Bacteria, size = 10)
    plt.yticks(rotation = 90)

    plt.show()