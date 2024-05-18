# Table of Contents

- [Background](#background)
- [Setup](#setup)
- [Methods](#methods)
  * [1. Exponential Smoothing](#1-exponential-smoothing)
    + [I. Training](#i-training)
    + [II. Testing](#ii-testing)
    + [III. More information](#iii-more-information)
  * [2. Regression (Exploratory)](#2-regression--exploratory-)
  * [3. Vector Similarity (Exploratory)](#3-vector-similarity--exploratory-)
- [Implementation](#implementation)
  * [1. Steps](#steps)
  * [2. Repo structure](#repo-structure)
  * [3. Parameters](#parameters)
- [Contact](#contact)

# Background
Automating the long-term demand forecast at a SKU/basename level using historical trends in customer behaviors. Solution provides three methods: exponential smoothing, gradient-descent based regressions and vector similarity to predict the percentage of demand on each SKU (per program family and customer) for a future quarter. Currently, the primary method is exponential smoothing. The other two methods are in an exploratory phase.  

# Setup
The goal is utilize this forecast tool for cq+2 onwards, where cq is the current quarter. The forecast for cq and cq+1 is to computed differently. A demand analyst would provide the total demand for for a program family in cq+2 and the tool would disaggregate this demand across all SKUs in the program family using the forecasting methods described below.

There are two cases based on the availablility of historical data.

Quarter to predict = q<br>
Minimum number of quarters needed for prediction (without legacy relationships) = n (6 is default)<br>
Number of quarters of available data = a<br>
A group is defined by a unique program family, interface, MLC/SLC and customer. Take group A for example, which comprises ADPRR, SATA, TLC drives for Amazon.<br> 
  
For group A,<br>

1. Case I:<br>
If ANY SKU/basename in the group has demand data for ALL (q - 1), (q - 2) … (q - n) quarters, i.e. a >= n for ANY SKU/basename<br>
Then use (q - 1), (q - 2) … (q - n) for forecast<br>
(For SKUs/basenames in this group that do not have demand data for all (q - 1), (q - 2) … (q - n) quarters, only use the subset of quarters that are available for the forecast. Use only a (< n) quarters in the computation. Do NOT reference any previous generation families)<br>

3. Case II:<br>
If NO basename in the group has demand data for ALL (q - 1), (q - 2) … (q - n), i.e. a <= n for ALL SKU/basenames<br>
Then use previous generation families and stitch the historicals according to the legacy program relationships the stakeholders provide<br>

Note: The code in this repository only addresses case I. The solution for case II has not been developed yet.

# Methods 

## 1. Exponential Smoothing 

Simple Exponential Smoothing can be interpreted as a weighted sum of the time-series values, wherein the weights are exponentially increasing (greater importance to future values in the time-series). The "alpha" value or the smoothing parameter lies between 0 and 1: the greater the value of alpha, the greater is the exponentially increasing nature of the weights. The formula is given below and as you can see it is recursive. The "alpha" value determines how much of the time-series history is used to forecast the next value. 

Learn more here: https://btsa.medium.com/introduction-to-exponential-smoothing-9c2d5909a714

### I. Training

The obvious follow-up question upon learning the formula for the exponential smoothing method is how to determine the best alpha value. The ExpSmoothing package: https://pypi.org/project/ExpSmoothing/ is used to train a exponential smoothing model using the following error metrics: 

| Error (Cost Function) | Parameter | Formula |
| ------------ | ------------------- | --------- |
| Mean Squared Error (MSE) |  ```mean squared error```  | <img src = "https://github.com/akomarla/ExpSmoothing/assets/124313756/a58bc3d7-6661-4995-825d-b031bd62016a" width = "30%" height = "30%"> <tr></tr> |
| Root Mean Squared Error (RMSE) |  ```root mean squared error```  | <img src = "https://github.com/akomarla/ExpSmoothing/assets/124313756/13106816-f256-4e74-ad06-b20470cc6f74" width = "35%" height = "35%"> <tr></tr> |
| Mean Absolute Error (MAE) |  ```mean absolute error```  | <img src = "https://github.com/akomarla/ExpSmoothing/assets/124313756/a5821e63-0020-4fa2-aea7-993ba6c6babe" width = "30%" height = "30%"> <tr></tr> |
| Mean Absolute Percentage Error (MAPE) |  ```mean absolute percentage error```  | <img src = "https://github.com/akomarla/ExpSmoothing/assets/124313756/4825f7e2-f0c6-4396-b27f-2333542f2d84" width = "35%" height = "35%"> <tr></tr> |

Where n represents the number of time-series in the data set. 

Learn more about the different cost functions here: https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/. 

### II. Testing

The ExpSmoothing package: https://pypi.org/project/ExpSmoothing/ is used to test an exponential smoothing model. In the testing step, a new set of time-series data is passed to the model to forecast future time-series values using the alpha value learned from the training step. The error from testing is the difference between the forecasted time-series values and the true future values of the time-series. 

### III. More information

See https://pypi.org/project/ExpSmoothing/ for more documentation on the method and implementation.

## 2. Regression (Exploratory) 

Use gradient descent to learn the optimal weights for a weighted average of a short-range time-series. The "optimal" weight vector will be one that minimizes the error between the forecasted values of the time-series and the true values. 

## 3. Vector Similarity (Exploratory) 

Learn areas of a time-series that are the best predictors of future values in the time-series. Rank the values in the time-series based on which values are nearest to the future value. The order [q_6, q_4, q_5, q_3, q_2, q_1] of importance can be represented as a vector [6, 4, 5, 3, 2, 1]. Use the cosine-similarity metric and unsupervised-learning to find clusters of weight vectors

<img src = "https://github.com/akomarla/automate_drive_demand_dissag/assets/124313756/f503a413-6a4f-4bc6-a736-d4f1f382385b"  width = "50%" height = "50%">

# Implementation 

The implementation is focused on the exponential smoothing method. For each SKU, a time-series of historical demand is extracted from the DMT database. The model then generates a forecast for each of these time-series in % form. The demand analyst provides the total demand forecast for a program family. The forecasted %s are applied and a final demand forecast value is generated for each SKU. 

## 1. Steps 

1. Read the DMT data and apply transformations to generate a set time-series that each represent one SKU
2. Replace all negative demand values with 0 (this modification is not an optional parameter)
3. Set a target quarter to forecast, ex: 'Q1 2024'
4. Set a number for the historical quarters to be used in the calculation, ex: 6
5. Set a error type to train the model, ex: 'mean absolute percentage error'
6. Train the exponential smoothing model. By default, the model trains on (tq - 1), where tq is the target quarter
7. Execute the model on a new set of training data
8. Write the results to an Excel file

## 2. Repo structure

```bash
├── code
│   ├── exp_smoothing
│   │   ├── singleft.py
│   │   ├── multft.py
│   │   ├── sec_ord.py
│   │   ├── config.py
│   ├── functions.py
│   ├── data_extraction.py
├── data
│   ├── output.xlsx
```

`singleft.py`: Produces a forecast for a single quarter in the future horizon, say 'Q2 2024'<br>
`multft.py`: Produces a forecast for multiple quarters in the future horizon, say 'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'<br>
`config.py`: Specify the quarters to forecast, database details, path to write the results<br>
`sec_ord.py`: Examining second order relationships across SKUs within a program family (ex: an increasing demand in high density SKUs in the ADPRRR family) <br>

## 3. Parameters 

For the config.py file:

| Data Type | Parameter |             Short Description                | Default Value |
| :--- | :--- | :----------------------------------------- | :--- |
| `str` | `q` | Target quarter to forecast | '2023Q3' |
| `int` | `ft_range` | Number of quarters to be used in the forecast computation | 6 |
| `str` | `train_error_type` | Error metric to use to train the model  | 'mean absolute percentage error' |
| `str` | `raw_demand_command` | Command to read raw demand data from the DMT database | null |
| `str` | `products_command` | Command to read the products data from the DMT database  | null |
| `str` | `server_name` | Server for DMT data | null |
| `str` | `database_name` | Database for DMT data  | 'demand' |
| `boolean` | `excel_output` | Specify whether to write the output to an Excel file or not  | True |
| `str` | `write_file_path` | Path where forecast outputs are written | null |

For the ExpSmoothing model and specifically the train() function:

| Data Type | Parameter |             Short Description                | Default Value |
| :--- | :--- | :----------------------------------------- | :--- |
| `list` | `train_data` | Time-series data to train the model | None |
| `list` | `train_true_values` | True future values of the training data | None |
| `str` | `error` | Error metric to use to train the model  | 'mean absolute percentage error' |
| `int` | `num_gen` | Number of future values in the time-series to generate | 1 |
| `boolean` | `remove_outliers` | Remove outliers in the time-series to train the model  | False |
| `str` | `how` | Specify whether to remove outliers using the IQR method: 'iqr' or just percentiles: 'percentile' | 'percentile' |
| `boolean` | `non_neg` | Remove negative values in the time-series to train the model  | False |
| `boolean` | `non_zero` | Remove zero values in the time-series to train the model  | False |

# Contact 
Contact Aparna Komarla (aparna.komarla@solidigm.com) with any questions!
