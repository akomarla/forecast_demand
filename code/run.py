# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:04:19 2023

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
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from functions import *
from base import *

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

# For a given quarter (in the past), get the true demand of all the previous quarters in the horizon

hist = gen_quart_hist_data(df = dmt_raw, 
                           sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                           quart_ft = ['2023Q2'], 
                           var_ft = 'cMrk_MGB', 
                           level_ft = 'Basename', 
                           quart_horizon = 10)

##############################################################################################################################################################

# For a given quarter, get all the forecasts made for the quarter (including the actuals)

# Target quarter to forecast
tg = TimeInstance(time_val = '2023Q2', time_type = 'year quarter')
# Historical forecast months
ext = [(0, 'dsf_1'), (3, 'dsf_2'), (6, 'jian_1'), (9, 'jian_2')]
data_dict = {}

# Write historical forecasts to dictionary
for count, label in ext:
    ym = tg.subtract_months(num = count, index = -1)
    jd_cycle = ym[0:4]+'_'+ym[4:6]+' JD'
    # Extract the data for the jd cycle and quarter
    df = extr_sql_data(command = "SELECT * FROM [demand].[dbo].[T_Demand_Archive] WHERE [Demand Cycle] = '"+jd_cycle+"' AND Quarter = '"+tg.year_quarter+"'", 
                       server_name = 'GOAPPS-SQL.CORP.NANDPS.COM,1433', 
                       database_name = 'demand')
    # Merge on product metadata
    df = df.merge(products, how = 'left', on = 'MM')
    # Calculate the MGB column
    df['cMrk_MGB'] = df['Qty (u)']*df['Mktg Density'].apply(clean_density_data)/1000000
    # Calculate the total demand per product family and percentage of demand for each basename
    df = sum_perc_calc(df, sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter', 'Demand Cycle'], 
                       sum_var = ['cMrk_MGB'], 
                       label = ['Quarter'], 
                       group_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family'])
    # Cleaning the extra white space on the interface column
    df['Interface'] = df['Interface'].str.rstrip()
    # Assign the df to a dictionary
    data_dict[label] = df

# Write actuals to the dictionary 
data_dict['actuals'] = sum_perc_calc(df = dmt_raw[dmt_raw['Quarter'] == tg.year_quarter], 
                                     sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                                     sum_var = ['cMrk_MGB'], 
                                     label = ['Quarter'], 
                                     group_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Quarter'])

# Write each of the dataframes - jian, dsf and actuals to Excel files
for label in data_dict.keys():
    data_dict[label].to_excel('C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/data/results/'+label+' '+tg.year_quarter+'.xlsx')

##############################################################################################################################################################

# Training the ES model 

# Generating the historical data
hist = gen_quart_hist_data(df = dmt_raw, 
                           sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                           quart_ft = ['2023Q1'],
                           var_ft = 'cMrk_MGB', 
                           level_ft = 'Basename', 
                           quart_horizon = 15)

# Test train split
train_hist, test_hist = train_test_split(hist.loc['Quarter % total'].reset_index(drop = True), test_size = 0.2)

# Quarter range for training
quart_ft = TimeInstance('2023Q1', 'year quarter')
quart_range = quart_ft.gen_quart_range(how = 'backward', num = 15)

# Remove any categorical or non-numeric columns
for i in quart_range: 
    train_hist = train_hist.drop(train_hist.index[train_hist[i] <= 0])

# Extract final training and true values (structure correctly)
train_hist_true = [[val] for val in train_hist[quart_range[-1]].tolist()]
train_hist = train_hist[quart_range[:-1]].values.tolist()
test_hist_true = [[val] for val in test_hist[quart_range[-1]].tolist()]
test_hist = test_hist[quart_range[:-1]].values.tolist()

# Train the model
es = ExpSmoothing()
es.train(train_hist, train_hist_true, 'mean absolute percentage error', 1)
# Test the model
ft, error = es.test(test_hist, test_hist_true)

###############################################################################

# Analyze and predict future values

# Getting the historical data
hist = gen_quart_hist_data(df = dmt_raw, 
                           sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Quarter'], 
                           quart_ft = ['2023Q2'],
                           var_ft = 'cMrk_MGB', 
                           level_ft = 'Basename', 
                           quart_horizon = 7).loc['Quarter % total']
# Example forecasts 
quart_ft = TimeInstance('2023Q1', 'year quarter')
quart_range = quart_ft.gen_quart_range(how = 'backward', num = 6)

ft_avg_non_neg = []
ft_avg = []
ft_exp_smooth = []
for i, row in hist.iterrows():
    ts = Analyze(values = row[quart_range].to_list(), labels = None)
    ft_avg_non_neg.append(ts.forecast(how = 'average non-neg'))
    ft_avg.append(ts.forecast(how = 'average'))
    ft_exp_smooth.append(ts.forecast(how = 'exponential smoothing', param = es.param, num_gen = 1)[0][0])

# Add columns with forecast and difference b/w true and forecasted values
hist['Average non-neg'] = ft_avg_non_neg
hist['Average non-neg (diff)'] = hist['2023Q2'] - hist['Average non-neg']

hist['Average'] = ft_avg
hist['Average (diff)'] = hist['2023Q2'] - hist['Average']

hist['Exponential smoothing forecast'] = ft_exp_smooth
hist['Exponential smoothing forecast (diff)'] = hist['2023Q2'] - hist['Exponential smoothing forecast']

hist = sum_perc_calc(df = hist, 
                     sel_vars = None, 
                     sum_var = ['Average non-neg', 'Average', 'Exponential smoothing forecast'],
                     label = ['Average non-neg', 'Average', 'Exponential smoothing forecast'], 
                     group_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family'])

# Write results to Excel
hist.to_excel('C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/data/results/test.xlsx')

###############################################################################

# hist_plus_ext = copy.deepcopy(hist_ext)

# for label in data_dict.keys():
#     if label != 'actuals':
#         df = data_dict[label]
#         df = df.rename({'Quarter % total': 'Quarter % total '+'('+label+')',
#                         'Demand Cycle': 'Demand Cycle '+'('+label+')'}, axis=1)
#         df = df[['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'Quarter % total '+'('+label+')', 'Demand Cycle '+'('+label+')']]
#         hist_plus_ext = pd.merge(hist_plus_ext, df, on = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename'], how = 'outer')

# # Write to excel
# hist_plus_ext.to_excel('C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/data/results/test2.xlsx', index = False)
        