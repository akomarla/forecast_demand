# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:05:28 2023

@author: akomarla
"""

import pandas as pd
import numpy as np
import logging
import os
import pyodbc as po
from datetime import datetime
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from functions import *

##############################################################################################################################################################

# Get the raw data

# Metadata on products
products = extr_sql_data(command = "SELECT * FROM [demand].[dbo].[T_Products]", 
                         server_name = 'GOAPPS-SQL.CORP.NANDPS.COM,1433', 
                         database_name = 'demand')
# Current cycle dmt data (contains actuals for past quarters and forecasts for current and future quarters)
dmt_raw = extr_sql_data(command = 'EXEC [demand].[dbo].[sp_Demand_Comp];', 
                        server_name = 'GOAPPS-SQL.CORP.NANDPS.COM,1433', 
                        database_name = 'demand')

##############################################################################################################################################################

# Training the ES model 

# Replacing negative values in raw data with 0s
num = dmt_raw._get_numeric_data()
num[num < 0] = 0

# Initialize all training and testing ts list
train = []
train_true = []
test = []
test_true = []

# Generate a training and testing set with multiple quarters as the target forecast
for quart in ['2021Q1', '2021Q2', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2']:
    # Generating the historical data
    hist = gen_quart_hist_data(df = dmt_raw, 
                               sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                               quart_ft = [quart],
                               var_ft = 'cMrk_MGB', 
                               level_ft = 'Basename', 
                               quart_horizon = 15).loc['Quarter % total']  
        
    # Removing any ts (historical and true value with NaNs)
    hist = hist._get_numeric_data()
    hist = hist[~np.isnan(hist).any(axis = 1)]
    
    # Test train split
    train_hist, test_hist = train_test_split(hist.reset_index(drop = True), test_size = 0.3)
    # Quarter range for training
    quart_ft = TimeInstance(quart, 'year quarter')
    quart_range = quart_ft.gen_quart_range(how = 'backward', num = 15)  
    # Extract final training and true values (structure correctly)
    train_hist_true = [val for val in train_hist[quart_range[-1]].tolist()]
    train_hist = train_hist[quart_range[:-1]].values.tolist()
    test_hist_true = [val for val in test_hist[quart_range[-1]].tolist()]
    test_hist = test_hist[quart_range[:-1]].values.tolist()
    
    # Complete training and testing dataset
    train.extend(train_hist)
    train_true.extend(train_hist_true)
    test.extend(test_hist)
    test_true.extend(test_hist_true)
    
# Find the quarter that is closest to the target quarter in each time-series
diff = []
q = []
for i in range(0, len(train)):
    ts_diff = []
    for j in range(0, len(train[i])):
        # Difference between each value in the time-series (0,...,n-1) and the true next value n
        d = (np.abs(train[i][j] - train_true[i])/train_true[i])*100
        ts_diff.append(d)
    min_diff = np.min(ts_diff)
    # Find the index of the value in the time-series that is closest to the true value
    if len(np.where(ts_diff == min_diff)[0]) != 1: 
        min_q = np.nan
    else: 
        min_q = np.where(ts_diff == np.min(ts_diff))[0][0]
    diff.append(min_diff)
    q.append(min_q)

##############################################################################################################################################################

# Histogram of quarter that is the best predictor of the target quarter

plt.hist(q)
plt.ylabel('count of ts')
plt.xlabel('q in ts w/ demand closest to that of target q')

##############################################################################################################################################################

# Get the order of quarters that best predict the target quarter

how = 'ascending'
q_ord = []
for i, ts in enumerate(train):
    # Difference of each value in the time-series with the true or target value
    diff = [(np.abs(val - train_true[i])/train_true[i])*100 for val in ts]
    if ~np.isnan(diff).any() and ~np.isinf(diff).any():
        # Order of the quarters closest to the target
        q_ord.append(list(np.argsort(diff)))

how = 'percentile'
q_ord = []
ranges = [[x,x+9] for x in range(0,100,10)]
for i, ts in enumerate(train):
    # % difference of each value in the time-series with the true or target value
    diff = [(np.abs(val - train_true[i])/train_true[i])*100 for val in ts]
    if ~np.isnan(diff).any() and ~np.isinf(diff).any():
        arg = []
        # Order of the quarters closest to the target
        for val in diff:
            match = False
            for i, rg in enumerate(ranges):
                if (val >= rg[0]) and (val <= rg[1]):
                    arg.append(i+1)
                    match = True
            if not match:
                arg.append(0.0001)
        q_ord.append(arg)
        
        
how = 'bell'

                    
##############################################################################################################################################################

# Find the quarter orders that are most similar to a specific order using cosine-similarity metric

a = [5,4,3,2,1,0]
count = 0
# Calculate cosine-similarity 
for b in q_ord:
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    if cos_sim >= 0.9:
        print(b)
        count += 1
        
##############################################################################################################################################################
        
# K-Means clustering 

x_norm = preprocessing.normalize(q_ord)
km2 = cluster.KMeans(n_clusters=5,init='random').fit(x_norm)
km2.cluster_centers_
km2.labels_

# Principal compenents analysis 

# Scaling the data
pca = PCA(n_components = 5)
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
Xt = pipe.fit_transform(q_ord)
# 3-D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], c = list(km2.labels_))
# 2-D plot
plot = plt.scatter(Xt[:,0], Xt[:,1], c = list(km2.labels_))
plt.show()