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
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from functions import *

class TimeSeries:
    def __init__(self, values, labels):
        """
        Parameters
        ----------
        values : list of floats
            list of float values representing the time-series, ex: [10, 15, 20,..]
        labels : list of strs
            list of string values representing the time-index of the time-series values, ex: [2023Q1, 2023Q2, 2023Q3, ...]

        Returns
        -------
        None.
        """
        self.values = values
        self.labels = labels
    
    def non_zero(self, inplace = False):
        """

        Parameters
        ----------
        inplace : boolean, optional
            specify whether to replace the time-series values with modified values or not. The default is False.

        Returns
        -------
        pv : list
            list of non-zero values in the time-series

        """
        # Initialize list of non-zero values
        nz = []
        for val in self.values:
            if val != 0:
                nz.append(val)
        
        # Replacing the original time-series with the modified one
        if inplace:
            self.values = nz
        # Retaining the original time-series and returning the modified one
        else:
            return nz
    
    def non_neg(self, inplace = False):
        """

        Parameters
        ----------
        inplace : boolean, optional
            specify whether to replace the time-series values with modified values or not. The default is False.

        Returns
        -------
        pv : list
            list of positive values in the time-series

        """
        # Initialize list of positive values
        pv = []
        for val in self.values:
            if val >= 0:
                pv.append(val)
        
        # Replacing the original time-series with the modified one
        if inplace:
            self.values = pv
        # Retaining the original time-series and returning the modified one
        else:
            return pv
        
    def clean(self, drop_na = True, inplace = False):
        """
        
        Parameters
        ----------
        drop_na : boolean, optional
            specify whether to remove NaN values from the time-series or not. The default is True.
        inplace : boolean, optional
            specify whether to replace the time-series values with modified values or not. The default is False.

        Returns
        -------
        wna : list
            list of values in the original time-series that are not NaN
        
        """
        # Check that time-series values are all integers or floats
        for val in self.values:
            if (type(val) == float) or (type(val) == int):
                pass
            else:
                print('Input data error. Value passed is not float or integer')
    
        # Dropping nan values in the time-series
        if drop_na:
            wna = [val for val in self.values if not np.isnan(val)]
        
        # Replacing the original time-series with the modified one
        if inplace:
            self.values = wna
        # Retaining the original time-series and returning the modified one
        else:
            return wna
        
    def check(self):
        # Check if labels and values are of the same length (if labels is a valid attribute)
        if self.labels:
            if len(self.values) != len(self.labels):
                print('Mismatch between labels and time-series values. Processing will be incorrect')
        
    def length(self):
        self.len = len(self.values)
        return self.len
    
    def plot(self):
        plt.plot(x = self.values, y = self.labels)
        plt.show()
        


class ExpSmoothing():
    def __init__(self):
        pass
    
    def es_model(self, ts, smoothing_level, num_gen):
        """

        Parameters
        ----------
        ts : list of floats
            ordered list of float values in a single time-series, ex: [1.2, 1.4, 1.8,...]
        smoothing_level : float
            weighting parameter for the exponential smoothing model
        num_gen : int
            number of future values of the time series to forecast, ex: 1, 2

        Returns
        -------
        ft : float
            future forecasted value of the time-series

        """
        
        # Run the model
        model = SimpleExpSmoothing(ts)
        fit = model.fit(smoothing_level)
        ft = fit.forecast(num_gen)
        return ft
        
    def train(self, train_data, train_true_values, error, num_gen):
        """

        Parameters
        ----------
        train_data : list of lists of floats
            time-series data to train the model, ex: [[1,2,3], [4,5,6],..]
        train_true_values : list of lists of floats
            true future values of the training data, ex: [[4,5], [7,8],...]
        error : string
            type of error to use to train the model, ex: 'mean square error', 'mean absolute error'
        num_gen : int
            number of future values in the time-series to generate, ex: 1

        Attributes
        -------
        param : dict
            contains parameters of the model that yield highest accuracy
            
        """
        # Assigning true data to instance
        self.train_data = train_data
        self.train_true_values = train_true_values
        # Assigning the number of values to forecast to instance
        self.num_gen = num_gen
        
        # Training by minimizing mean squared error 
        e_dict = {}
        for alpha in np.arange(0.05,1,0.05):
            ft_val = []
            for ts in self.train_data:
                try:
                    # Set up baseline model for a single time-series
                    _ts = TimeSeries(values = ts, labels = None)
                    # Make modifications to the time series
                    _ts.clean(inplace = True, drop_na = True)
                    # Run the model
                    ft = self.es_model(ts = _ts.values, smoothing_level = alpha, num_gen = self.num_gen)
                    ft_val.append(ft.tolist())
                except:
                    ft_val.append([np.nan]*self.num_gen)
            
            # Finding the indices of non-NaN values 
            mask = ~pd.isna(ft_val) & ~pd.isna(self.train_true_values)
            # Training by minimizing mean squared error 
            if error == 'mean squared error':
                try:
                    e_dict[alpha] = metrics.mean_squared_error(np.array(ft_val)[mask], np.array(self.train_true_values)[mask])
                except:
                    pass
            # Training by minimizing mean absolute error 
            elif error == 'mean absolute error':
                try:
                    e_dict[alpha] = metrics.mean_absolute_error(np.array(ft_val)[mask], np.array(self.train_true_values)[mask])
                except:
                    pass
            # Training by minimizing the root mean squared error
            elif error == 'root mean squared error':
                try:
                    e_dict[alpha] = metrics.mean_squared_error(np.array(ft_val)[mask], np.array(self.train_true_values)[mask], squared = False)
                except:
                    pass
            # Training by minimizing the percentage error
            elif error == 'mean absolute percentage error':
                try:
                    e_dict[alpha] = metrics.mean_absolute_percentage_error(np.array(ft_val)[mask], np.array(self.train_true_values)[mask])
                except:
                    pass 
            # If no correct error type is passed
            else:
                print('Error type specified is not recognized. Please pass a valid error type to use for the optimization')
  
        # Store best alpha value in dictionary - minimum error
        self.param = {}
        self.param['alpha'] = min(e_dict, key = e_dict.get)
        # Store the chosen error type
        self.error = error
        # Store the training error associated with the best alpha
        self.train_error = min(e_dict.values())
            
    def test(self, test_data, test_true_values, error = None, num_gen = None, param = None):
        """

        Parameters
        ----------
        test_data : list of lists of floats
            time-series data to test the model, ex: [[1,2,3], [4,5,6],..]
        test_true_values : list of lists of floats
            true future values of the testing data, ex: [[4,5], [7,8],...]
        error : str, optional
            type of error to test the model, ex: 'mean square error', 'mean absolute error'
            default is None
            ignore to test the model using the training input itself
        num_gen : int, optional
            number of future values in the time-series to generate, ex: 1
            default is None
            ignore to test the model using the training input itself
        param : dict, optional
            contains parameters to test the model
            default is None
            ignore to test the model using the optimal parameters from the training process

        Returns
        -------
        ft_val : list of lists of floats
            future values of the time-series, ex: [[4,5], [7,8],...]
        test_error : float
            error of the forecast compared to true future values

        """
        # Assigning test data to instance
        self.test_data = test_data
        self.test_true_values = test_true_values
        
        # Select which input to use
        # If no parameters and inputs are passed by user, use training input and optimal parameter from training process (stored as attributes in the instance)
        if not param: 
            param = self.param
        if not num_gen:
            num_gen = self.num_gen
        if not error:
            error = self.error
            
        # Initialize list of forecasted values
        ft_val = []
        for ts in self.test_data:
            try:
                # Set up baseline model for a single time-series
                _ts = TimeSeries(values = ts, labels = None)
                # Make modifications to the time series
                _ts.clean(inplace = True, drop_na = True)
                # Run the model
                ft = self.es_model(ts = _ts.values, smoothing_level = param['alpha'], num_gen = num_gen)
                ft_val.append(ft.tolist())
            except:
                ft_val.append([np.nan]*num_gen)

        # Finding the indices of non-NaN values
        mask = ~pd.isna(ft_val) & ~pd.isna(self.test_true_values)
        # Calculate the mean squared error 
        if error == 'mean squared error':
            try:
                self.test_error = metrics.mean_squared_error(np.array(ft_val)[mask], np.array(self.test_true_values)[mask])
            except:
                pass
        # Calculate the mean absolute error 
        elif error == 'mean absolute error':
            try:
                self.test_error = metrics.mean_absolute_error(np.array(ft_val)[mask], np.array(self.test_true_values)[mask])
            except:
                pass
        # Calculate the root mean squared error 
        elif error == 'root mean squared error':
            try:
                self.test_error = metrics.mean_squared_error(np.array(ft_val)[mask], np.array(self.test_true_values)[mask], squared = False)
            except:
                pass
        # Training by minimizing the percentage error
        elif error == 'mean absolute percentage error':
            try:
                self.test_error = metrics.mean_absolute_percentage_error(np.array(ft_val)[mask], np.array(self.test_true_values)[mask])
            except:
                pass 
        # If no correct error type is passed
        else:
            print('Error type specified is not recognized. Please pass a valid error type to use for the optimization')
            
        return ft_val, self.test_error
    
    def predict(self, data, num_gen = None, param = None):
        """

        Parameters
        ----------
        data : list of lists of floats
            time-series data to forecast, ex: [[1,2,3], [4,5,6],..]
        num_gen : int, optional
            number of future values in the time-series to generate, ex: 1
            default is None
            ignore to forecast values using the training input itself
        param :  dict, optional
            contains parameters to test the model
            default is None
            ignore to forecast values using the optimal parameters from the training process

        Returns
        -------
        ft_val : list of lists of floats
            future values of the time-series, ex: [[4,5], [7,8],...]

        """
        # Select which input to use
        # If no parameters and inputs are passed by user, use training input and optimal parameter from training process (stored as attributes in the instance)
        if not param: 
            param = self.param
        if not num_gen:
            num_gen = self.num_gen
            
        ft_val = []
        for ts in data: 
            try:
                # Set up baseline model for a single time-series
                _ts = TimeSeries(values = ts, labels = None)
                # Make modifications to the time series
                _ts.clean(inplace = True, drop_na = True)
                ft = self.es_model(ts = _ts.values, smoothing_level = param['alpha'], num_gen = num_gen)
                ft_val.append(ft.tolist())
            except:
                ft_val.append([np.nan]*num_gen)
        return ft_val
        
    
        
class Analyze(TimeSeries, ExpSmoothing):
    def __init__(self, values, labels):
        """

        Parameters
        ----------
        values : list of floats
            inherited from TimeSeries class
        time_labels : list of strings
            inherited from TimeSeries class
            
        Returns
        -------
        None.

        """
        super().__init__(values, labels)
        
    def average(self, non_neg = False):
        """

        Parameters
        ----------
        non_neg : boolean, optional
            specify whether to compute average with or without negative values. The default is False.

        Returns
        -------
        avg : float
            mean or average of the values

        """
        # Extract the non negative values and then calculate the average
        if non_neg:
            try:
                nn = self.non_neg(inplace = False)
                return sum(nn)/len(nn)
            except:
                return np.nan
        else:
            try:
                return sum(self.values)/len(self.values)
            except:
                return np.nan
        
    def forecast(self, how, num_gen = None, param = None):
        """

        Parameters
        ----------
        how : str
            method to use to forecast, ex: 'average', average non-neg', 'exponential smoothing'
        num_gen : int
            number of future values to generate using exponential smoothing forecast, ex: 1,2. Default is None
        param : dict, optional
            parameters needed for the exponential smoothing forecast, ex: {'alpha': 0.5}. Default is None

        Returns
        -------
        forecast : dict
            forecast values with the method, ex: {'average': [0.1, 0.2], 'average non-neg': [0.2, 0.3]}

        """
        # Check that the length of values and labels are the same
        self.check()
        # Remove NaN values in the time-series and replace with non-NaN modified list
        self.clean(inplace = True)
        
        # Generate forecast using following methods
        if how == 'average':
            return self.average(non_neg = False)
        elif how == 'average non-neg':
            return self.average(non_neg = True)
        elif how == 'exponential smoothing':
            return self.predict(data = [self.values], num_gen = num_gen, param = param)



class TimeInstance:
    
    def __init__(self, time_val, time_type, desc = None):
        """
        Parameters
        ----------
        time_val : str
            raw input time value, ex: '2023_03 JD', 
            '2023Q1', 'Q12023', '052023', '202305' (month must always starts with leading zero for 1-9)
        time_type : str
            description of the time value format, ex: 'year quarter', 'year month'
        desc : str
            additional description of the time value, ex: 'jd cycle'
            
        Attributes
        ----------
        month : int or list
            value of the month, ex: month of Q1 is [1,2,3]
        year : int
            value of the year, ex: 2024
        quarter : str
            value of the quarter, ex: 'Q1'
        year_quarter : str
            value of the year and quarter combined, ex: '2023Q1'
        year_month : str
            value of the year and month combined, ex: '202302'
        """
        # Assign all attributes to the instance
        self.time_type = time_type
        self.time_val = time_val
        self.desc = desc
        
        # Assign remaining attributes: quarter, month, year, etc. based on input format
        
        # If time value is JD cycle; expected input is of the form: '2023_03 XY'
        if (self.time_type == 'year month') & (self.desc == 'jd cycle'):
            # Assign month value using input
            self.month = int(time_val[5:7])
            # Assign year value using input
            self.year = int(time_val[0:4])
            # Assign quarter value using month: 01, 02,.. 12
            if self.month in [1,2,3]:
                self.quarter = 'Q1'
            elif self.month in [4,5,6]:
                self.quarter = 'Q2'
            elif self.month in [7,8,9]:
                self.quarter = 'Q3'
            elif self.month in [10,11,12]:
                self.quarter = 'Q4'
            else:
                self.quarter = None
            # Assign quarter and year combined value
            self.year_quarter = str(self.year)+self.quarter
            # Assign month and year combined value
            self.year_month = str(self.year)+'{:02}'.format(self.month)
        
        # If time value is simple year and month with no additional category; expected input is of the form: '202301'
        elif (self.time_type == 'year month') & (self.desc == None):
            # Assign month value using input
            self.month = int(time_val[4:6])
            # Assign year value using input
            self.year = int(time_val[0:4])
            # Assign quarter value using month: 01, 02,.. 12
            if self.month in [1,2,3]:
                self.quarter = 'Q1'
            elif self.month in [4,5,6]:
                self.quarter = 'Q2'
            elif self.month in [7,8,9]:
                self.quarter = 'Q3'
            elif self.month in [10,11,12]:
                self.quarter = 'Q4'
            else:
                self.quarter = None
            # Assign quarter and year combined value
            self.year_quarter = str(self.year)+self.quarter
            # Assign month and year combined value
            self.year_month = str(self.year)+'{:02}'.format(self.month)
        
        # If time value is simple month and year with no additional category; expected input is of the form: '022023'
        elif self.time_type == 'month year':
            # Assign month value using input
            self.month = int(time_val[0:2])
            # Assign year value using input
            self.year = int(time_val[2:6])
            # Assign quarter value using month: 01, 02,.. 12
            if self.month in [1,2,3]:
                self.quarter = 'Q1'
            elif self.month in [4,5,6]:
                self.quarter = 'Q2'
            elif self.month in [7,8,9]:
                self.quarter = 'Q3'
            elif self.month in [10,11,12]:
                self.quarter = 'Q4'
            else:
                self.quarter = None
            # Assign quarter and year combined value
            self.year_quarter = str(self.year)+self.quarter
            # Assign month and year combined value
            self.year_month = str(self.year)+'{:02}'.format(self.month)
            
        # If time value is standard year and quarter; expected input is something of the form: '2023Q1'
        elif self.time_type == 'year quarter':
            # Assign year value using input
            self.year = int(time_val[0:4])
            # Assign quarter value using input
            self.quarter = time_val[4:6]
            # Assign month value(s) using quarter: Q1, Q2...
            if self.quarter == 'Q1':
                self.month = [1,2,3]
            elif self.quarter == 'Q2':
                self.month = [4,5,6]
            elif self.quarter == 'Q3':
                self.month = [7,8,9]
            elif self.quarter == 'Q4':
                 self.month = [10,11,12]
            else:
                self.month = None
            # Assign quarter and year combined value
            self.year_quarter = time_val
            # Assign month and year combined value
            self.year_month = [str(self.year)+'{:02}'.format(month) for month in self.month]
                
        # If time value is quarter and year; expected input is something of the form: 'Q12023'
        elif self.time_type == 'quarter year':
            # Assign quarter value using input
            self.quarter = time_val[0:2]
            # Assign year value using input
            self.year = int(time_val[2:6])
            # Assign month value(s) using quarter: Q1, Q2...
            if self.quarter == 'Q1':
                self.month = [1,2,3]
            elif self.quarter == 'Q2':
                self.month = [4,5,6]
            elif self.quarter == 'Q3':
                self.month = [7,8,9]
            elif self.quarter == 'Q4':
                 self.month = [10,11,12]
            else:
                self.month = None
            # Assign quarter and year combined value
            self.year_quarter = str(self.year)+self.quarter
            # Assign month and year combined value
            self.year_month = [str(self.year)+'{:02}'.format(month) for month in self.month]

        else:
            print('The format of the time value passed is not recognized. Follow one of the accepted formats and ensure that all inputs are passed correctly')
            self.month = None
            self.year = None
            self.quarter = None
            self.year_quarter = None
            self.year_month = None
            
    def gen_month_range(self, how, num, index = None):
        """
        Parameters
        ----------
        how : str
            indicates how to generate the range, ex: 'forward', 'backward'
        num : int
            count of month to generate, ex: 6
        index: int
            index of the month to use as the starting or ending point in the range when month is a list, ex: 1
        
        Returns
        -------
        fill : list
            year and month values, ex: ['202101', '202102', '202103'....]
        """
        # Ensure the required input is passed, if so, continue executing the function
        # If month and year values are not null
        if self.month and self.year:
            # If month and year are single values (not lists) and integers
            if ((type(self.month) == int) and (type(self.year) == int)):
                pass
            elif ((type(self.month) == list) and (type(self.year) == int)):
                start_m = self.month[index]
                end_m = self.month[index]
                pass
            # Terminate the function if conditions are not met
            else:
                return None
        # Terminate the function if conditions are not met
        else:
            return None
        
        # Generate ranges under different conditions 
        # When start and number of future quarters to generate is given
        if (how == 'forward') & (num is not None): 
            # Extract years and quarters for start 
            start_y, start_m = self.year, start_m
            # Initialize list of months to generate
            fill = []
            # Initialize month and year value to start incrementing from
            y, m = start_y, start_m
            count = 0 
            # Continue till all months are generated
            while count <= num:
                if m <= 12:
                    fill.append(str(y)+'{:02}'.format(m))
                    count += 1
                    m += 1
                # Reset quarter and increase year 
                elif m >= 12:
                    y += 1
                    m = 1
            return fill
        
        # When end and number of previous quarters to generate is given             
        elif (how == 'backward') & (num is not None):
            # Extract years and quarters for start 
            end_y, end_m = self.year, end_m
            # Initialize list of months to generate
            fill = []
            # Initialize month and year value to start decreasing from
            y, m = end_y, end_m
            count = 0 
            # Continue till all months are generated
            while count <= num:
                if (m <= 12) & (m > 0):
                    fill.append(str(y)+'{:02}'.format(m))
                    count += 1
                    m -= 1
                # Reset month and decrease year 
                elif m <= 0:
                    y -= 1
                    m = 12       
            # Reverse the list since we are looking in the backward direction
            fill.reverse()
            return fill 
        
        else:
            print('Input data insufficient. No months generated. Fix input data and re-run')
    
    def gen_quart_range(self, how, num):
        """
        Parameters
        ----------
        how : str
            indicates how to generate the range, ex: 'forward', 'backward'
        num : int
            count of quarters to generate, ex: 6

        Returns
        -------
        fill : list
            year and quarter values, ex: ['2021Q1', '2021Q2', '2021Q3'....]
        """
        # Ensure the required input is passed, if so, continue executing the function
        # If month and year values are not null
        if self.quarter and self.year:
            try:
                # If quarter and year are single values (not lists) and correct data type
                if ((type(self.quarter) == str) and (type(self.year) == int)):
                    pass
            # Terminate the function if conditions are not met
            except:
                return None
        # Terminate the function if conditions are not met
        else:
            return None
        
        # Generate ranges under different conditions 
        
        # When start and number of future quarters to generate is given
        if (how == 'forward') & (num is not None): 
            # Extract years and quarters for start 
            start_y, start_q = self.year, self.quarter
            # Initialize list of quarters to generate
            fill = []
            # Initialize quarter and year value to start incrementing from
            y, q = start_y, int(start_q[-1])
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
        elif (how == 'backward') & (num is not None):
            # Extract years and quarters for end
            end_y, end_q = self.year, self.quarter
            # Initialize list of quarters to generate
            fill = []
            # Initialize quarter and year value to start decreasing from
            y, q = end_y, int(end_q[-1])
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
            # Reverse the list since we are looking in the backward direction
            fill.reverse()
            return fill 
        
        else:
            print('Input data insufficient. No quarters generated. Fix input data and re-run')
        
    def add_quarters(self, num):
        """
        Parameters
        ----------
        num : int
            count of quarters to generate, ex: 6

        Returns
        -------
        fill : list
            year and quarter values, ex: ['2021Q1', '2021Q2', '2021Q3'....]
        """
        return self.gen_quart_range(how = 'forward', num = num)[-1]
    
    def subtract_quarters(self, num):
        """
        Parameters
        ----------
        num : int
            count of quarters to generate, ex: 6

        Returns
        -------
        fill : list
            year and quarter values, ex: ['2021Q4', '2021Q3', '2021Q2'....]
        """
        return self.gen_quart_range(how = 'backward', num = num)[-1]
    
    def add_months(self, num, index = None):
        """
        Parameters
        ----------
        num : int
            count of months to generate, ex: 6
        index: int
            index of the month to use as the starting or ending point in the range 
            when month is a list, ex: 1
            
        Returns
        -------
        fill : list
            year and month values, ex: ['202101', '202102', '202103'....]
        """
        return self.gen_month_range(how = 'forward', num = num, index = index)[-1]
    
    def subtract_months(self, num, index = None):
        """
        Parameters
        ----------
        num : int
            count of months to generate, ex: 6
        index: int
            index of the month to use as the starting or ending point in the range when month is a list, ex: 1

        Returns
        -------
        fill : list
            year and month values, ex: ['202104', '202103', '202102'....]
        """
        return self.gen_month_range(how = 'backward', num = num, index = index)[-1]