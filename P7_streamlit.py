import torch
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import shap

from P7_functions import final_preprocessing
from P7_functions_streamlit import *
from sklearn.linear_model import LogisticRegression
from PIL import Image

# -----------------------------Define caching functions ---------------------------------
@st.cache
def load_model(filename):
    load_clf = pickle.load(open(filename, 'rb'))
    return load_clf

@st.cache
def get_datas():
    
    # Loading matrices
    x = pd.read_csv('./Clean_datas/real_x.csv')
    x.index = x['SK_ID_CURR']
    
    y = pd.read_csv('./Clean_datas/real_y.csv')
    y.index = x['SK_ID_CURR']
    
    x.drop('SK_ID_CURR', axis=1, inplace=True)
    y.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Loading imputer and preprocessor
    imputer = pickle.load(open('./Models/imputer.pkl', 'rb'))
    preprocessor = pickle.load(open('./Models/preprocessor.pkl', 'rb'))
    
    return x, y, preprocessor, imputer

@st.cache
def load_descriptions():
    descriptions_df = pd.read_csv('./Clean_datas/var_description.csv')
    return descriptions_df
    
# -----------------------------Define containers ----------------------------------------
header = st.container()
prediction_container = st.container()
explanation = st.container()
visualization = st.container()

# -----------------------------Load datas -----------------------------------------------

# Load saved classification model
load_clf = load_model('./Models/final_model.pkl')

# Load our data set
x, y, preprocessor, imputer = get_datas()

# -----------------------------Define sidebar -------------------------------------------

# Upload datas : two options, select a customer in existing dataset, or upload files for a new one 
choose_option = st.sidebar.selectbox('Data for prediction', ['Existing customer', 'Upload datas'])

if choose_option == 'Existing customer':
    customer_list = list(x.index)
    customer = st.sidebar.selectbox('Customer selection', customer_list)
    selected_customer = x.loc[customer]

else:
    
    # Upload files
    upload_counter = 0

    uploaded_file_1 = st.sidebar.file_uploader("Upload your input application_data CSV file", type=["csv"])
    if uploaded_file_1:
        uploaded_application_data = pd.read_csv(uploaded_file_1)
        upload_counter += 1

    uploaded_file_2 = st.sidebar.file_uploader("Upload your input bureau_balance CSV file", type=["csv"])
    if uploaded_file_2:
        uploaded_bureau_balance = pd.read_csv(uploaded_file_2)
        upload_counter += 1

    uploaded_file_3 = st.sidebar.file_uploader("Upload your input bureau CSV file", type=["csv"])
    if uploaded_file_3:
        uploaded_bureau = pd.read_csv(uploaded_file_3)
        upload_counter += 1

    uploaded_file_4 = st.sidebar.file_uploader("Upload your input previous_application CSV file", type=["csv"])
    if uploaded_file_4:
        uploaded_previous_application = pd.read_csv(uploaded_file_4)
        upload_counter += 1

    uploaded_file_5 = st.sidebar.file_uploader("Upload your input credit_card_balance CSV file", type=["csv"])
    if uploaded_file_5:
        uploaded_credit_card_balance = pd.read_csv(uploaded_file_5)
        upload_counter += 1

    uploaded_file_6 = st.sidebar.file_uploader("Upload your input POS_CASH_balance CSV file", type=["csv"])
    if uploaded_file_6:
        uploaded_POS_CASH_balance = pd.read_csv(uploaded_file_6)
        upload_counter += 1

    uploaded_file_7 = st.sidebar.file_uploader("Upload your input installments_payments CSV file", type=["csv"])
    if uploaded_file_7:
        uploaded_installments_payments = pd.read_csv(uploaded_file_7)
        upload_counter += 1

    # If all files have been uploaded
    if upload_counter == 7: 
        
        # Cleaning, feature engineering and merge
        uploaded_datas, _ = preprocess_joint(
            uploaded_application_data, uploaded_bureau_balance, uploaded_bureau, uploaded_previous_application, 
            uploaded_credit_card_balance, uploaded_POS_CASH_balance, uploaded_installments_payments, imputer=imputer
        )
        
        # Preprocessing for model (standardization, encoding, imputation)
        x_upload, y_upload, _ = final_preprocessing(uploaded_datas, is_balance=False, 
                                                    existing_preprocessor=preprocessor, is_existing_cols=True, full_cols=x)
       
        # Choose a customer
        customer_list = list(x_upload.index)
        customer = st.sidebar.selectbox('Customer selection', customer_list)
        selected_customer = x_upload.loc[customer]
        
    else:
        st.sidebar.write("All files haven't been uploaded yet")


# -----------------------------Dashboard ------------------------------------------------

# Presentation of our application
with header:
    st.write("""
    # Customer loan repayment capacity prediction application
    This app predicts the ability of a customers to repay a loan based on personnal informations and previous loans payments
    """)

# Make predictions
threshold = 0.5
probas = load_clf.predict_proba(selected_customer.values.reshape(1, -1))[:,1]
predictions = (probas >= threshold).astype(int)

# Display predictions
with prediction_container:
    st.header("Predictions of selected customer")
    
    predic_col, proba_col = st.columns(2)
    
    # Display if loan accepted or refused with an image
    predic_col.subheader('Prediction')
    outputs = np.array(['Accepted','Refused'])
    predic_col.write(outputs[predictions][0])
    predic_col.write("")
    predic_col.write("")
    predic_col.write("")
    predic_col.write("")
    predic_col.write("")
    predic_col.write("")
    predic_col.write("")
    
    if outputs[predictions][0] == 'Accepted':
        image = Image.open('./Images/Tick.png')
        predic_col.image(image)
    else:
        image = Image.open('./Images/Cross.png')
        predic_col.image(image)       

    # Display probability with a gauge
    proba_col.subheader('Prediction Probability')
    proba_col.write("{:.3f}".format(probas[0]))
    
    gauge = {
        'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'steps' : [{'range': [0, threshold], 'color': "lightgreen"}, {'range': [threshold, 1], 'color': "lightcoral"}],
        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}
    }
    
    fig = go.Figure(
        go.Indicator(
            mode = "gauge+number",
            value = probas[0],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probability"},
            gauge = gauge
        ),
        layout={'height' : 300, 'width' : 300}
    )

    proba_col.plotly_chart(fig)

# Explain features determining prediction
explainer = shap.LinearExplainer(load_clf, x)

with explanation:
    st.header("Global features importance (training dataset)")
    shap_values = explainer.shap_values(x)
    fig = plt.figure()
    plot = shap.summary_plot(shap_values, x, plot_type="bar", color="dodgerblue")
    st.pyplot(fig)
    
    # Display descriptions of features in chart
    descriptions_df = load_descriptions()
    
    is_description_global = st.checkbox('Display description of variables', key='global')

    if is_description_global:
    
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
    
    # Display descriptions of features in chart
    
    is_description_local = st.checkbox('Display description of variables', key='local')
    
    if is_description_local:

        shap_df = pd.Series(abs(shap_values), index=x.columns)
        top_features = shap_df.sort_values(ascending=False)[:10].index
        selected_df = descriptions_df[descriptions_df['Variable'].isin(top_features)]

        for feat in top_features:
            row = selected_df[selected_df['Variable'] == feat]
            st.write("Variable : {}".format(row['Variable'].values[0]))
            st.write("DataFrame : {}".format(row['Var_Dataframe'].values[0]))
            st.write("Description : {}".format(row['Var_Description'].values[0]))
            st.write("-----------------------------------------------------------------")
            
with visualization:
    # Get back list of local and global top features
    shap_values = explainer.shap_values(x)
    shap_df = pd.DataFrame(abs(shap_values), columns=x.columns)
    top_global_features = shap_df.mean(axis=0).sort_values(ascending=False)[:20].index
    
    shap_values = explainer.shap_values(selected_customer)
    shap_df = pd.Series(abs(shap_values), index=x.columns)
    top_local_features = shap_df.sort_values(ascending=False)[:10].index
    
    features_diff = list(set(top_local_features) - set(top_global_features))
    top_global_features = list(top_global_features)
    for feat in features_diff:
        top_global_features.append(feat)
        
    st.header("Top features visualization (training dataset)")
    
    feat_1_col, feat_2_col = st.columns(2)
    
    feat_1 = feat_1_col.selectbox('Feature 1', top_global_features)
    feat_1_col.write("Distribution of {}".format(feat_1))
    plt.figure()
    sns.displot(x=x[feat_1], hue=y['TARGET'], kind="kde", palette=['g', 'r'])
    feat_1_col.pyplot(plt.gcf())
    
    
    feat_2 = feat_2_col.selectbox('Feature 2', top_global_features)
    feat_2_col.write("Distribution of {}".format(feat_2))
    plt.figure()
    sns.displot(x=x[feat_2], hue=y['TARGET'], kind="kde", palette=['g', 'r'])
    feat_2_col.pyplot(plt.gcf())
    
    st.write("Correlation between 2 selected features")
    plt.figure()
    sns.scatterplot(x=x[feat_1], y=x[feat_2], hue=y['TARGET'], palette=['g', 'r'])
    st.pyplot(plt.gcf())
    