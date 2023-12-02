# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:26:20 2023

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
from ExpSmoothing import *
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
