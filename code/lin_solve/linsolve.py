# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:15:29 2023

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
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

# Generating the historical data
hist = gen_quart_hist_data(df = dmt_raw, 
                           sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                           quart_ft = ['2023Q2'],
                           var_ft = 'cMrk_MGB', 
                           level_ft = 'Basename', 
                           quart_horizon = 6).loc['Quarter % total']
  

##############################################################################################################################################################

# Removing any ts (historical and true value with NaNs)
hist = hist._get_numeric_data()
hist = hist[~np.isnan(hist).any(axis = 1)]

# Test train split
train_hist, test_hist = train_test_split(hist.reset_index(drop = True), test_size = 0.3)

# Quarter range for training
quart_ft = TimeInstance('2023Q2', 'year quarter')
quart_range = quart_ft.gen_quart_range(how = 'backward', num = 6)

# Extract final training and true values (structure correctly)
train_hist_true = [val for val in train_hist[quart_range[-1]].tolist()]
train_hist = train_hist[quart_range[:-1]].values.tolist()
test_hist_true = [val for val in test_hist[quart_range[-1]].tolist()]
test_hist = test_hist[quart_range[:-1]].values.tolist()

##############################################################################################################################################################

# Visualizing trends

diff = []
q = []
for i in range(0, len(train_hist)):
    ts_diff = []
    for j in range(0, len(train_hist[i])):
        d = (np.abs(train_hist[i][j] - train_hist_true[i])/train_hist_true[i])*100
        ts_diff.append(d)
    min_diff = np.min(ts_diff)
    if len(np.where(ts_diff == np.min(ts_diff))[0]) != 1: 
        min_q = np.nan
    else: 
        min_q = np.where(ts_diff == np.min(ts_diff))[0][0]
    diff.append(min_diff)
    q.append(min_q)

plt.hist(q)
plt.ylabel('count of ts')
plt.xlabel('q in ts w/ demand closest to that of target q')

##############################################################################################################################################################

regr = SGDRegressor(max_iter=1000, tol=1e-3, fit_intercept = False)
regr.fit(X = train_hist, y = train_hist_true)
regr.predict(X = test_hist)
regr.intercept_
regr.coef_

# Always scale the input. The most convenient way is to use a pipeline.
reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3, fit_intercept = False))
reg.fit(X = train_hist, y = train_hist_true)
reg.predict(X = test_hist)
