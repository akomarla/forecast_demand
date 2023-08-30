# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:04:19 2023

@author: akomarla
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from functions import *

###############################################################################

# Read demand data
dmt_data = pd.read_excel('C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/Data/DMT-Ritem Pivot.xlsx', sheet_name = 'dmt data')

# Initialize list for historicals and quarter range
agg_hist = []
quart_range = gen_quarters(['2021Q3', '2026Q1'])
for key, group in dmt_data.groupby(by = ['CUSTOMER_NAME', 'Family', 'Basename', 'MM']):
    if '2023Q3' in group['Quarter'].unique():
        # Storing customer, family and quarters available
        hist = [key[0], key[1], key[2], key[3], group['Quarter'].unique()]
        for quart in quart_range:
            if quart in group['Quarter'].unique():
                hist.append('x')
            else:
                hist.append(np.nan)
        agg_hist.append(hist)

# Convert to dataframe and write to Excel sheet
cust_fam_hist = pd.DataFrame(agg_hist, columns = ['Customer', 'Program family', 'Basename', 'MM', 'Quarters forecasted']+quart_range).sort_values(by = list(reversed(quart_range)))
cust_fam_hist.to_excel('C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/Data/Results/cust_fam_hist.xlsx', index = False)

###############################################################################

indp_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family']
dissag_var = ['Basename', 'MM']
dep_var = ['cMrk_MGB']
time_var = ['Quarter']
ft = '2024Q1'
quart_range = gen_quarters(start = None, end = ft, num = 6)
hist_pres = 0 
hist_abs = 0 

# Add quarter sum and percentage of demand
dmt_data_mod = pd.DataFrame()
dmt_gps_indp_var = dmt_data[indp_var + dissag_var + time_var + dep_var].groupby(by = indp_var, dropna = False)
for key, group in tqdm(dmt_gps_indp_var):
    for quarter in group['Quarter'].unique():
        quarter_sum = group[group['Quarter'] == quarter][dep_var].sum()
        group.loc[group['Quarter'] == quarter, 'Quarter '+dep_var[0]+' total'] = quarter_sum[0]
        group.loc[group['Quarter'] == quarter, 'Quarter % '+dep_var[0]+' total'] = (group[group['Quarter'] == quarter][dep_var]*100/quarter_sum).values.tolist()
    dmt_data_mod = pd.concat([dmt_data_mod, group])
            
###############################################################################

indp_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family']
dissag_var = ['Basename', 'MM']
dep_var = ['cMrk_MGB']
time_var = ['Quarter']
ft = '2024Q1'
quart_range = gen_quarters(start = None, end = ft, num = 6)
hist_pres = 0 
hist_abs = 0 

# Identifying groups with enough historical data for the quarter we want to forecast
for key, group in dmt_data_mod[indp_var + dissag_var + time_var + dep_var].groupby(by = indp_var + dissag_var):
    if ft in group['Quarter'].unique():
        check = True
        for quart in quart_range:
            if quart not in group['Quarter'].unique():
                check = False
        if check:
            print(key)
            hist_pres += 1
        else:
            hist_abs += 1           
       
###############################################################################

    