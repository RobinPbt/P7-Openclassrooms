import torch
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

from P7_functions import final_preprocessing_2
from P7_functions import display_descriptions
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

@st.cache
def load_descriptions():
    descriptions_df = pd.read_csv('./Clean_datas/var_description.csv')
    return descriptions_df

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

# Select if we show descriptions
is_description = st.sidebar.selectbox('Show description', [False, True])

# Display predictions
with prediction_container:
    st.header("Predictions of selected customer")
    
    predic_col, proba_col = st.columns(2)
    
    predic_col.subheader('Prediction')
    outputs = np.array(['Accepted','Refused'])
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
    
    # Display descriptions of features in chart
    descriptions_df = load_descriptions()
    
    if is_description:
    
        shap_df = pd.DataFrame(abs(shap_values), columns=x.columns)
        top_features = shap_df.mean(axis=0).sort_values(ascending=False)[:20].index
        selected_df = descriptions_df[descriptions_df['Variable'].isin(top_features)]

        for feat in top_features:
            row = selected_df[selected_df['Variable'] == feat]
            st.write("Variable : {}".format(row['Variable'].values[0]))
            st.write("DataFrame : {}".format(row['Var_Dataframe'].values[0]))
            st.write("Description : {}".format(row['Var_Description'].values[0]))
            st.write("-----------------------------------------------------------------")
    
    
    st.header("Local features importance")
    shap_values = explainer.shap_values(selected_customer)
    fig = plt.figure()
    plot = shap.bar_plot(shap_values, features=selected_customer, max_display=10)
    st.pyplot(fig)
    
    if is_description:

        shap_df = pd.Series(abs(shap_values), index=x.columns)
        top_features = shap_df.sort_values(ascending=False)[:10].index
        selected_df = descriptions_df[descriptions_df['Variable'].isin(top_features)]

        for feat in top_features:
            row = selected_df[selected_df['Variable'] == feat]
            st.write("Variable : {}".format(row['Variable'].values[0]))
            st.write("DataFrame : {}".format(row['Var_Dataframe'].values[0]))
            st.write("Description : {}".format(row['Var_Description'].values[0]))
            st.write("-----------------------------------------------------------------")