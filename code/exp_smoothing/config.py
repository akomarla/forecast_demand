# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:08:53 2023

@author: akomarla
"""

# Quarter to be forecasted
q = '2023Q3'
ft_range = 6
train_error_type = 'mean absolute percentage error'
excel_output = True
write_path = 'C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/data/results/test.xlsx'

# Reading data from DMT database
products_command = "SELECT * FROM [demand].[dbo].[T_Products]"
raw_demand_command = 'EXEC [demand].[dbo].[sp_Demand_Comp];'

server_name = 'GOAPPS-SQL.CORP.NANDPS.COM,1433'
database_name = 'demand'

# Number of future quarters to forecast for (relevant to the multiple quarter forecast)
num_ft = 5