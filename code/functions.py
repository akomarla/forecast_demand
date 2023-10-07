# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 23:34:42 2023

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
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from base import *

def extr_sql_data(command, server_name, database_name, csv_output = None, write_file_path = None, excel_output = None, table_name = None):
    """ 
    
    Parameters
    ----------
    command : 
        str of SQL statement (query, stored procedure, etc) to execute, ex: 'EXEC [demand].[dbo].[sp_Demand_Comp];'
    server_name : 
        str of server name, ex: 'GOAPPS-SQL.CORP.NANDPS.COM,1433'
    database_name : 
        str of database name, ex: 'demand'
    table_name : 
        str of table name inside database, ex: 'demand_pegatron'
    csv_output : 
        boolean to or to not return .csv file, ex: True, False, None
    excel_output : 
        boolean to or to not return .xlsx file, ex: True, False, None
    write_file_path : 
        str of path to store output
    
    Returns 
    -------
    df : pandas dataframe, excel output
        dataframe and file with the extracted data from the SQL databse
        
    """
    try:
        # Make connection to the SQL server and database
        conn = po.connect('Driver={SQL Server};'
                         'Server='+server_name+';'
                         'Database='+database_name+';'
                         'Trusted_Connection=yes;')
        # Execute the specified command
        df = pd.read_sql(command, conn)    
        
        # Write output to csv or excel
        if csv_output:
            df.to_csv(write_file_path)
        
        elif excel_output:
            df.to_excel(write_file_path)
    
    except:
        print('Unable to execute SQL statement. Please ensure that the server and database names are accurate.')
        df = None
        
    return df 


def clean_density_data(den):
    """
    
    Parameters
    ----------
    den : str
        density value with units, ex: '512GB'
        
    Returns 
    -------
    den : int
        numerical value of the density input (without units)
    
    """
    try:
        den = int(den[0:-2])
    except:
        den = np.nan
    return den


def gen_quart_hist_data(df, sel_vars, quart_ft, var_ft, level_ft, fill_zero = False, quart_horizon = 6):
    """

    Parameters
    ----------
    df : pandas dataframe
        raw dataframe with raw historical data
    sel_vars : list
        list of desired columns to be included in output
    quart_ft : list of strs
        list of quarter and year values to be forecasted, ex: ['2023Q2, '2023Q1']
    quart_horizon : int
        number of quarters of historicals to be generated, default is 6
    var_ft : str
        name of variable to be forecasted, ex: 'cMrk_MGB'
    level_ft : str
        drill-down variable level (or SKU) to forecast, ex: 'Family', 'Basename'
    fill_zero : boolean, optional
        specify whether to fill missing values in the output with 0s, ex: True, False or None
        
    Returns
    -------
    hist : pandas dataframe
        dataframe with row-wise historical data 
        (columns are quarters and row values correspond to the level and forecast variables)

    """
    def single_horizon(ft):
        # Intializing the quarter range for generating historical data
        quart_range = TimeInstance(ft, 'year quarter').gen_quart_range(how = 'backward', num = quart_horizon)
        
        if sel_vars:
            gps = df[sel_vars].groupby(by = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family'], dropna = False)
        else:
            gps = df.groupby(by = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family'], dropna = False)
        
        # Initialize dataframe with historical data
        sh = pd.DataFrame()
        # Identifying groups with enough historical data for the quarter we want to forecast
        for _, g in gps:
            check = True
            for q in quart_range:
                # Check if program family has all of the quarters in the horizon (any SKU level)
                if q not in g['Quarter'].unique():
                    check = False
            # Only generate historical data if the family has all quarters in the horizon 
            if check:
                # Calculate the percentage and total per unit at a group level
                g = sum_perc_calc(df = g, 
                                  sel_vars = None, 
                                  sum_var = [var_ft], 
                                  label = ['Quarter'], 
                                  group_var = ['CUSTOMER_NAME', 'Interface', 'MLC/SLC', 'Family', 'Quarter'])
                # Get all quarters for a basename or family
                l_gps = g.groupby(by = level_ft, dropna = False)
                for _, lg in l_gps:
                    # Get the numeric columns to be summed
                    cols = lg._get_numeric_data().columns.tolist()+['Quarter']
                    # Extract quarters in the range, perform summing and transpose so that time-series is in a row
                    # Sum the quarter total, % and other numerical columns by quarter
                    ext = lg[lg['Quarter'].isin(quart_range)][cols].groupby('Quarter').sum().transpose()
                    # Get the non-numeric categorical columns
                    cols = [c for c in lg.columns if c not in ['Quarter', var_ft]+list(ext.index)]
                    # Update results with the non-numeric values
                    ext[cols] = [lg[c].unique()[0] for c in cols]
                    sh = pd.concat([sh, ext])         
        
        # Fill all missing values with 0s -> for all quarters with no data
        if fill_zero:
            sh = sh.fillna(0)
        
        return sh 
    
    # Generate historical data for multiple target quarters and concatenate the data
    def mult_horizon(quart_ft):
        mh = pd.DataFrame()
        # Combine historical data for each target quarter at a time
        for ft in quart_ft:
            mh = pd.concat([single_horizon(ft), mh])
        return mh
    
    return mult_horizon(quart_ft)


def sum_perc_calc(df, sel_vars, sum_var, label, group_var):
    """

    Parameters
    ----------
    df : pandas dataframe
        input data
    sel_vars : list
        list of variables of interest
    sum_var : list of strs
        names of variable to use for calculation - sum and %
    label : list of strs
        names of new column/row to store calculations - should correspond to sum variables in a 1:1 fashion
    group_var : str or list
        name of variable or list of variables to group the data - str if one variable only, list of strs if more than one variable

    Returns
    -------
    df_mod : pandas dataframe
        output data with sum and % of variables specified

    """
    def single_var(df, sv, l):
        # Initialize modified dataframe
        df_mod = pd.DataFrame()
        # Select a subset of the dataframe and group
        if sel_vars:
            gps = df[sel_vars].groupby(by = group_var, dropna = False)
        else:
            gps = df.groupby(by = group_var, dropna = False)
        
        # Calculate the total variable value and percentage per unit
        for _, g in tqdm(gps):
            g[l+' total'] = g[sv].sum()
            g[l+' % total'] = (g[sv]/g[sv].sum())*100
            df_mod = pd.concat([df_mod, g])
    
        return df_mod
    
    def mult_var(df, sum_var, label):
        # Iterate over all variables to be summed
        #df_mod = pd.DataFrame()
        for sv, l in zip(sum_var, label):
            df = single_var(df, sv, l)
    
        return df
    
    return mult_var(df, sum_var, label)
    