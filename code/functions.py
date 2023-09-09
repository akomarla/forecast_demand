# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:23:50 2023

@author: akomarla
"""

import pandas as pd
import numpy as np
import logging
import os
import pyodbc as po
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt


def gen_quarters(start, end, num = None):
    """ 
    :param start: str of starting year and quarter, ex: '2021Q1'
    :param end: str of ending year and quarter, ex: '2021Q1'
    :param num: int of number of quarters to generate from start or end, ex: 6
    
    returns list of year and quarter values ['2021Q1', '2021Q2', '2021Q3'....]
    """
    # When time range is given
    if (start is not None) & (end is not None):
        # Extract years and quarters for start 
        start_y, start_q = int(start[:4]), int(start[-1])
        # Initialize list of quarters to generate
        fill = []
        # Initialize quarter and year value to start incrementing from
        y, q = start_y, start_q
        # Continue till end quarter and range are reached
        while str(y)+'Q'+str(q) != end:
            if q <= 4:
                fill.append(str(y)+'Q'+str(q))
                q += 1
            # Reset quarter and increase year 
            elif q >= 5:
                y += 1
                q = 1
        return fill
    
    # When start and number of future quarters to generate is given
    elif (start is not None) & (num is not None): 
        # Extract years and quarters for start 
        start_y, start_q = int(start[:4]), int(start[-1])
        # Initialize list of quarters to generate
        fill = []
        # Initialize quarter and year value to start incrementing from
        y, q = start_y, start_q
        count = 0 
        # Continue till all quarters are generated
        while count <= num:
            if q <= 4:
                fill.append(str(y)+'Q'+str(q))
                count += 1
                q += 1
            # Reset quarter and increase year 
            elif q >= 5:
                y += 1
                q = 1
        return fill
    
    # When end and number of previous quarters to generate is given             
    elif (end is not None) & (num is not None):
        # Extract years and quarters for end
        end_y, end_q = int(end[:4]), int(end[-1])
        # Initialize list of quarters to generate
        fill = []
        # Initialize quarter and year value to start decreasing from
        y, q = end_y, end_q
        count = 0 
        # Continue till all quarters are generated
        while count <= num:
            if (q <= 4) & (q > 0):
                fill.append(str(y)+'Q'+str(q))
                count += 1
                q -= 1
            # Reset quarter and decrease year 
            elif q <= 0:
                y -= 1
                q = 4
        return fill 
    
    else:
        print('Input data insufficient. No quarters generated. Fix input data and re-run')
        
        
        
def extr_sql_data(command, server_name, database_name, 
                  csv_output = None, write_file_path = None, excel_output = None, table_name = None):
    """ 
    :param command: str of SQL statement (query, stored procedure, etc) to execute, ex: 'EXEC [demand].[dbo].[sp_Demand_Comp];'
    :param server_name: str of server name, ex: 'GOAPPS-SQL.CORP.NANDPS.COM,1433'
    :param database_name: str of database name, ex: 'demand'
    :param table_name: str of table name inside database, ex: 'demand_pegatron'
    
    returns pandas dataframe 
    """
    try:
        # Make connection to the SQL server and database
        conn = po.connect('Driver={SQL Server};'
                         'Server='+server_name+';'
                         'Database='+database_name+';'
                         'Trusted_Connection=yes;')
        # Execute the specified command
        df = pd.read_sql(command, conn)    
    
    except:
        print('Unable to execute SQL statement. Please ensure that the server and database names are accurate.')
    
    if csv_output:
        df.to_csv(write_file_path)
    
    elif excel_output:
        df.to_excel(write_file_path)
        
    return df 


def compute_avg(data, non_neg = False):
    """
    :param data: one-dimensional list of values
    
    returns float of average of non negative values in data
    """
    if non_neg:
        non_neg_vals = []
        for val in data:
            if val >= 0:
                non_neg_vals.append(val)
        return sum(non_neg_vals)/len(non_neg_vals)
    else:
        return sum(data)/len(data)
    
    
def ts_exp_smoothing(data, num_gen, alpha, non_neg = True):
    """
    :param data: one-dimensional list of values
    :param num_gen: int of number of forecasts to generate
    :param alpha: float of smoothing parameter for model
    :param non_neg: boolean specifies whether to use only positive values in the ts forecast
        
    returns list of forecasts
    """
    if non_neg: 
        # Get non-negative data 
        non_neg_vals = []
        for val in data:
            if val >= 0:
                non_neg_vals.append(val)
        # Run the model on non negative data only
        model = SimpleExpSmoothing(non_neg_vals)
        fit = model.fit(smoothing_level = alpha)
        ft = fit.forecast(num_gen)
    else:
        # Run the model on data as passed in the function
        model = SimpleExpSmoothing(data)
        fit = model.fit(smoothing_level = alpha)
        ft = fit.forecast(num_gen)
    return list(ft)
    