import torch
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

from P7_functions import final_preprocessing_2
from sklearn.linear_model import LogisticRegression

# Define caching for loading model and dataset
@st.cache
def load_model(filename):
    load_clf = pickle.load(open(filename, 'rb'))
    return load_clf

@st.cache
def get_datas():
    x, y = final_preprocessing_2()
    return x, y

# Function to plot shap charts
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
# Define our containers
header = st.container()
prediction_container = st.container()
explanation = st.container()

# Presentation of our application
with header:
    st.write("""
    # Customer loan repayment capacity prediction application
    This app predicts the ability of a customers to repay a loan based on personnal informations and previous loans payments
    """)

# Load saved classification model
load_clf = load_model('./Models/final_model.pkl')

# Load our data set
x, y = get_datas()

# Select a customer
customer_list = x.index
customer = st.sidebar.selectbox('customer', customer_list)
selected_customer = x.loc[customer]

# Make predictions
threshold = 0.5
probas = load_clf.predict_proba(selected_customer.values.reshape(1, -1))[:,1]
predictions = (probas >= threshold).astype(int)

# Display predictions
with prediction_container:
    st.header("Predictions of selected customer")
    
    predic_col, proba_col = st.columns(2)
    
    predic_col.subheader('Prediction')
    outputs = np.array(['No default','Default'])
    predic_col.write(outputs[predictions])

    proba_col.subheader('Prediction Probability')
    proba_col.write(probas)

# Explain features determining prediction
explainer = shap.LinearExplainer(load_clf, x)

with explanation:
    st.header("Global features importance")
    shap_values = explainer.shap_values(x)
    fig = plt.figure()
    plot = shap.summary_plot(shap_values, x, plot_type="bar", color="dodgerblue")
    st.pyplot(fig)  

    st.header("Local features importance")
    shap_values = explainer.shap_values(selected_customer)
    fig = plt.figure()
    plot = shap.bar_plot(shap_values, features=selected_customer, max_display=10)
    st.pyplot(fig)