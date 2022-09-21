import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from flask import Flask, render_template, url_for, request, jsonify
from preprocess_functions import *
from utils import *
from P7_functions import final_preprocessing

import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/api/predict-existing/', methods=['GET'])
def predict_existing():
    
    if 'id_customer' in request.args:
        
        customer = request.args['id_customer']
        
        # Load saved classification model
        load_clf = pickle.load(open('../Models/final_model.pkl', 'rb'))
        
        # Load datas of customer
        customer_features = find_customer_features(customer)
        
        # Make predictions
        threshold = 0.5
        probas = load_clf.predict_proba(customer_features.values.reshape(1, -1))[:,1]
        predictions = (probas >= threshold).astype(int)
        
        probas = float(probas[0])
        predictions = int(predictions[0])
        
        return jsonify({
            'status' : 'ok',
            'datas': {
                'prediction' : predictions,
                'proba' : probas
            }
        })
         
    else:
        return jsonify({
            'status': 'error',
            'message': 'You must specify parameter id_customer'
        }), 500

    
@app.route('/api/predict-new/', methods = ['POST'])
def predict_new():
    
    # Load model, imputer and preprocessor
    imputer = pickle.load(open('../Models/imputer.pkl', 'rb'))
    preprocessor = pickle.load(open('../Models/preprocessor.pkl', 'rb'))
    load_clf = pickle.load(open('../Models/final_model.pkl', 'rb'))
    
    # Load file
    file_1 = request.files['data']
    x = pd.read_csv(file_1)
    x.index = x['SK_ID_CURR']
    x.drop('SK_ID_CURR', axis=1, inplace=True)  

    # Make predictions
    threshold = 0.5
    probas = load_clf.predict_proba(x)[:,1]
    predictions = (probas >= threshold).astype(int)

    probas = float(probas[0])
    predictions = int(predictions[0])
    
    return jsonify({
    'status' : 'ok',
    'datas': {
        'prediction' : predictions,
        'proba' : probas
    }
})
    
if __name__ == "__main__":
    app.run(debug=True)


