# -*- coding: utf-8 -*-

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
from ExpSmoothing import *
import config
import sys

# Getting the functions file from one directory up
os.chdir('..')
sys.path.insert(1, os.getcwd())
from functions import *

##############################################################################################################################################################

# Get the raw data

# Metadata on products
products = extr_sql_data(command = config.products_command, 
                         server_name = config.server_name, 
                         database_name = config.database_name)
# Current cycle dmt data (contains actuals for past quarters and forecasts for current and future quarters)
dmt_raw = extr_sql_data(command = config.raw_demand_command, 
                        server_name = config.server_name, 
                        database_name = config.database_name)

##############################################################################################################################################################

# Initialize the target quarter for the forecast

target_q = TimeInstance(config.q, 'year quarter')

##############################################################################################################################################################

# Training the ES model 

# Replacing negative values in raw data with 0s
num = dmt_raw._get_numeric_data()
num[num < 0] = 0

# Generating the historical data
hist = gen_quart_hist_data(df = dmt_raw, 
                           sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                           quart_ft = [target_q.subtract_quarters(num = 1)],
                           var_ft = 'cMrk_MGB', 
                           level_ft = 'Basename', 
                           level_count = 6,
                           fill_zero = False,
                           quart_horizon = config.ft_range)

# Test train split
train_hist, test_hist = train_test_split(hist.loc['Quarter % total'].reset_index(drop = True), test_size = 0.4)

# Quarter range for training
quart_ft = TimeInstance(target_q.subtract_quarters(num = 1), 'year quarter')
quart_range = quart_ft.gen_quart_range(how = 'backward', num = config.ft_range)

# Extract final training and true values (structure correctly)
train_hist_true = [[val] for val in train_hist[quart_range[-1]].tolist()]
train_hist = train_hist[quart_range[:-1]].values.tolist()
test_hist_true = [[val] for val in test_hist[quart_range[-1]].tolist()]
test_hist = test_hist[quart_range[:-1]].values.tolist()

# Train the model
es = ExpSmoothing()
es.train(train_data = train_hist, 
         train_true_values = train_hist_true, 
         error = config.train_error_type, 
         num_gen = 1, 
         remove_outliers = True, 
         non_neg = True, 
         non_zero = True)

# Test the model
ft, error = es.test(test_data = test_hist, 
                    test_true_values = test_hist_true, 
                    remove_outliers = True, 
                    non_neg = True, 
                    non_zero = False)

##############################################################################################################################################################

# Analyze and predict future values

# Getting the historical data
hist = gen_quart_hist_data(df = dmt_raw, 
                           sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                           quart_ft = [target_q.time_val],
                           var_ft = 'cMrk_MGB', 
                           level_ft = 'Basename', 
                           level_count = 6,
                           fill_zero = False,
                           quart_horizon = config.ft_range).loc['Quarter % total']
# Example forecasts 
quart_ft = TimeInstance(target_q.subtract_quarters(num = 1), 'year quarter')
quart_range = quart_ft.gen_quart_range(how = 'backward', num = config.ft_range-1)

ft_avg_non_neg = []
ft_avg = []
ft_exp_smooth = []
for i, row in hist.iterrows():
    ts = Analyze(values = row[quart_range].to_list(), labels = None)
    ft_avg.append(ts.forecast(how = 'average'))
    ft_exp_smooth.append(ts.forecast(how = 'exponential smoothing', 
                                     param = es.param, 
                                     remove_outliers = True, 
                                     outliers_how = 'percentile', 
                                     non_neg = True, 
                                     non_zero = False,
                                     num_gen = 1)[0][0])

# Add columns with forecast 
hist['Average'] = ft_avg
hist['Exponential smoothing forecast'] = ft_exp_smooth

# Find the percentage of the true demand and forecasted demand for each method
hist = sum_perc_calc(df = hist, 
                     sel_vars = None, 
                     sum_var = ['Average', 'Exponential smoothing forecast'],
                     label = ['Average', 'Exponential smoothing forecast'], 
                     group_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family'])

# Add columns with forecast and difference b/w true and forecasted values
hist['Average (diff)'] = hist[target_q.time_val] - hist['Average % total']
hist['Exponential smoothing forecast (diff)'] = hist[target_q.time_val] - hist['Exponential smoothing forecast % total']

# If excel output is requested write to an output file
if config.excel_output:
    # Write results to Excel
    hist.to_excel(config.write_path)

##############################################################################################################################################################
