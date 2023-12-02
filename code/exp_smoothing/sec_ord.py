# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:58:13 2023

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
import scipy.stats
from ExpSmoothing import *
from functions import *
import config

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

# Initialize the target quarter for the forecast

target_q = TimeInstance(config.q, 'year quarter')

##############################################################################################################################################################

# Trends in demand based on density of SSD

# Replacing negative values in raw data with 0s
num = dmt_raw._get_numeric_data()
num[num < 0] = 0

# Generating the historical data
hist = gen_quart_hist_data(df = dmt_raw, 
                           sel_vars = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Basename', 'cMrk_MGB', 'Mktg Density', 'Quarter'], 
                           quart_ft = [target_q.subtract_quarters(num = 1)],
                           var_ft = 'cMrk_MGB', 
                           level_ft = 'Basename', 
                           quart_horizon = config.ft_range)     

# Format density column
hist['Mktg Density Val'] = hist['Mktg Density'].apply(lambda x: x.replace('GB', ''))
quart_range = target_q.gen_quart_range(how = 'backward', num = config.ft_range+1)[:-1]

slope = []
for i, row in hist.iterrows():
    # Demand values
    x = row[quart_range].to_list()
    # Remove nan values from time-series
    x = np.array(x)[~np.isnan(x)]
    # Calculate the slope of the linear regression only if the time-series has more than 5 values
    if len([val for val in x if val != 0]) >= 5:
        # Generate y values
        y = range(0, len(x))
        try: 
            # Linear regression
            m, c, r, p, err = scipy.stats.linregress(x, y)
            slope.append(m)
        except:
            slope.append(np.nan)
    else:
        slope.append(np.nan)
# Update with slopes per density
hist['Density Trend'] = slope

# Group slope and density values to check correlation
for den, df in hist.loc['Quarter % total'].groupby(by = 'Mktg Density Val'):
    print(den)
    print(df[df['Density Trend'].isna() == False]['Density Trend'])
    
##############################################################################################################################################################

# If excel output is requested write to an output file
if config.excel_output:
    # Write results to Excel
    hist.to_excel(config.write_path)
    
##############################################################################################################################################################
