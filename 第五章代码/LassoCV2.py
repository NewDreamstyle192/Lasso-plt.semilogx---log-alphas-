# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:24:41 2019

@author: 28137
"""

import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
from math import sqrt, log

'''
target_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = urllib.request.urlopen(target_url).
xlist = []
names = []
labels = []
firstline = True

for line in data:
    if firstline:
        names = line.strip().split(";")
        firstline = False
    else:
        row = line.strip().split(";")
        labels.append(float(row[-1]))
        row.pop
        floatrow = [float(i) for i in row]
        xlist.append(floatrow)
'''
file = open(r'D:\the treasure of CYY\Python.ML\data\winequality-red.csv', encoding='utf-8')
data = pd.read_csv(file, sep=';')
names = data.columns
labels = [float(num) for num in list(data.iloc[:, -1].values)]
data.drop(axis=1, columns='quality', inplace=True)
xlist = data.values
nrows = len(xlist)
ncols = len(xlist[1])

for i in range(nrows):
    rowdata = [float(xlist[i][k]) for k in range(ncols)]
    xlist[i] = rowdata

# 先求出每列的均值和标准差，然后进行标准化
xMeans = []
xSD = []

for k in range(ncols):
    # 求每一列均值
    mean_k = sum([xlist[i][k] for i in range(nrows)]) / nrows
    xMeans.append(mean_k)
    # 归一化
    colDiff = [(xlist[i][k] - mean_k) for i in range(nrows)]
    # 标准差
    Sq = sum([colDiff[i] * colDiff[i] for i in range(nrows)]) / nrows
    xSD_k = sqrt(Sq)
    xSD.append(xSD_k)

# 标准化样本值和标签值
xNormalized = []
for i in range(nrows):
    rowdata = [(xlist[i][k] - xMeans[k]) / xSD[k] for k in range(ncols)]
    xNormalized.append(rowdata)

labelsMean = sum(labels) / nrows
labelsDiff = [(labels[i] - labelsMean) for i in range(nrows)]
labelsSD = sqrt(sum([(labelsDiff[i] * labelsDiff[i]) for i in range(nrows)]) / nrows)

labelsNormalized = [(labels[i] - labelsMean) / labelsSD for i in range(nrows)]

x1 = np.array(xlist)
x2 = np.array(xNormalized)
y1 = np.array(labels)
y2 = np.array(labelsNormalized)

alphas, coefs, _ = linear_model.lasso_path(x2, y2, n_alphas=100, return_models=False)
alphas_log = [log(num) for num in alphas if num > 0] # 这里的改变
plt.plot(alphas_log, coefs.T)
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.axis('tight')

ax = plt.gca()
ax.invert_xaxis()
plt.show()

nattr, nalpha = coefs.shape
