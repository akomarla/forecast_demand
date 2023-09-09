# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:18:27 2023

@author: akomarla
"""

from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = "C:/Users/akomarla/OneDrive - NANDPS/Desktop/Repos/gbl_ops_data_analytics.forecasting.automation.demand_dissag/Data/Results"
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
 
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')
 
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        # Clear all session variables
        session.clear()
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
        return render_template('index2.html')
    return render_template("index.html")

@app.route('/input', methods=['GET', 'POST'])
def userInput():
    # User input
    if request.method == 'POST':
        program_family = request.form.get('program_family')
        session['program_family'] = program_family
        quarter = request.form.get('quarter')
        session['quarter'] = quarter
        customer = request.form.get('customer')
        session['customer'] = customer
        print(program_family, quarter, customer)
    return render_template("input.html")
    
 
@app.route('/show_results')
def showResults():
    # Uploaded file path
    data_file_path = session.get('uploaded_data_file_path', None)
    program_family = session.get('program_family', None)
    quarter = session.get('quarter', None)
    customer = session.get('customer', None)
    # Read Excel
    uploaded_df = pd.read_excel(data_file_path)
    # Make selections 
    sel_df = uploaded_df[(uploaded_df['Family'].str.lower() == program_family.lower()) & (uploaded_df['CUSTOMER_NAME'].str.lower() == customer.lower()) & (uploaded_df['Quarter'].str.lower() == quarter.lower())]
    return render_template('show_results.html', data_var = sel_df.head().to_html())
 
 
if __name__ == '__main__':
    app.run(debug=True)