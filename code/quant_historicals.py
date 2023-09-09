# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:04:19 2023

@author: akomarla
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from functions import *

###############################################################################

# Read demand data
dmt_data = pd.read_excel('C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/data/input/DMT-Ritem Pivot.xlsx', sheet_name = 'dmt data')
dmt_data = extr_sql_data(command = 'EXEC [demand].[dbo].[sp_Demand_Comp];', 
                         server_name = 'GOAPPS-SQL.CORP.NANDPS.COM,1433', 
                         database_name = 'demand')
  
###############################################################################

# Set the parameters
indp_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family']
dissag_var = 'Basename'
dep_var = 'cMrk_MGB'
time_var = 'Quarter'
ft = '2024Q1'
quart_range = gen_quarters(start = None, end = ft, num = 6)
hist_pres = 0 
hist_abs = 0 
hist_ext = pd.DataFrame()

###############################################################################

# Add quarter sum and percentage of demand
dmt_data_mod = pd.DataFrame()
fam_gps = dmt_data[indp_var + [dissag_var] + [time_var] + [dep_var]].groupby(by = indp_var, dropna = False)
for key, fg in tqdm(fam_gps):
    for quarter in fg[time_var].unique():
        quarter_sum = fg[fg[time_var] == quarter][dep_var].sum()
        fg.loc[fg[time_var] == quarter, 'Quarter '+dep_var+' total'] = quarter_sum
        fg.loc[fg[time_var] == quarter, 'Quarter % '+dep_var+' total'] = (fg[fg[time_var] == quarter][dep_var]*100/quarter_sum).values.tolist()
    dmt_data_mod = pd.concat([dmt_data_mod, fg])
            
###############################################################################

# Identifying groups with enough historical data for the quarter we want to forecast
fam_gps = dmt_data_mod[indp_var + [dissag_var] + [time_var] + [dep_var] + ['Quarter % '+dep_var+' total']].groupby(by = indp_var, dropna = False)
for key, fg in tqdm(fam_gps):
    if ft in fg[time_var].unique():
        check = True
        for quart in quart_range:
            # Check if program family has all of the quarters in the horizon (any basename)
            if quart not in fg[time_var].unique():
                check = False
        if check:
            hist_pres += 1
            bname_gps = fg.groupby(by = [dissag_var], dropna = False)
            for key, bg in tqdm(bname_gps):
                ex = bg[bg[time_var].isin(quart_range)].sort_values(by = time_var)[[time_var, 'Quarter % '+dep_var+' total']].groupby(time_var).sum()
                ex = ex.transpose()
                ex['Basename'] = bg['Basename'].unique()
                ex['CUSTOMER_NAME'] = bg['CUSTOMER_NAME'].unique()
                ex['Interface'] = bg['Interface'].unique()
                ex['MLC/SLC'] = bg['MLC/SLC'].unique()
                ex['Family'] = bg['Family'].unique()
                hist_ext = pd.concat([hist_ext, ex])
        else:
            hist_abs += 1           

# Fill all missing values with 0s -> for all quarters with no forecasts
hist_ext = hist_ext.fillna(0)

###############################################################################

# Example forecasts 
ft_avg_non_neg = []
ft_avg = []
ft_exp_smooth = []
for row in tqdm(range(0, len(hist_ext))):
    ord_quart_range = list(reversed(quart_range))
    data = hist_ext.iloc[row][ord_quart_range].to_list()
    ft_avg_non_neg.append(compute_avg(data = data, non_neg = True))
    ft_avg.append(compute_avg(data = data, non_neg = False))
    ft_exp_smooth.append(ts_exp_smoothing(data = data, num_gen = 1, alpha = 0.5)[0])

# Add columns with forecast and difference b/w true and forecasted values
hist_ext['Average non-neg'] = ft_avg_non_neg
hist_ext['Average non-neg (diff)'] = hist_ext[ft] - hist_ext['Average non-neg']

hist_ext['Average'] = ft_avg
hist_ext['Average (diff)'] = hist_ext[ft] - hist_ext['Average']

hist_ext['Exponential smoothing forecast'] = ft_exp_smooth
hist_ext['Exponential smoothing forecast (diff)'] = hist_ext[ft] - hist_ext['Exponential smoothing forecast']

# Write to excel
hist_ext.to_excel('C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/data/results/test.xlsx')

###############################################################################
