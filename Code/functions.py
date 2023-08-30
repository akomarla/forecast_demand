# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:23:50 2023

@author: akomarla
"""

def gen_quarters(start, end, num = None):
    """ 
    param: start -> string of starting year and quarter, ex: '2021Q1'
    param: end -> string of ending year and quarter, ex: '2021Q1'
    param: num -> int of number of quarters to gen from start, ex: 6
    returns: list of year and quarter values ['2021Q1', '2021Q2', '2021Q3'....]
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
        