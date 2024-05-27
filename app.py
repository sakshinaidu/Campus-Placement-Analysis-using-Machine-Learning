import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\sandi\OneDrive\Desktop\final project\archive (5).zip")
    return df

data = load_data()

# Preprocess the data
def preprocess_data(data):
    # Encoding categorical features
    data_encoded = pd.get_dummies(data, columns=['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation'])
    return data_encoded

data_encoded = preprocess_data(data)

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['M', 'F'])
    ssc_p = st.sidebar.slider('SSC Percentage', 0, 100, 50)
    ssc_b = st.sidebar.selectbox('SSC Board', ['Central', 'Others'])
    hsc_p = st.sidebar.slider('HSC Percentage', 0, 100, 50)
    hsc_b = st.sidebar.selectbox('HSC Board', ['Central', 'Others'])
    hsc_s = st.sidebar.selectbox('HSC Specialization', ['Commerce', 'Science', 'Arts'])
    degree_p = st.sidebar.slider('Degree Percentage', 0, 100, 50)
    degree_t = st.sidebar.selectbox('Degree Type', ['Sci&Tech', 'Comm&Mgmt', 'Others'])
    workex = st.sidebar.selectbox('Work Experience', ['Yes', 'No'])
    etest_p = st.sidebar.slider('Etest Percentage', 0, 100, 50)
    specialisation = st.sidebar.selectbox('MBA Specialisation', ['Mkt&HR', 'Mkt&Fin'])
    mba_p = st.sidebar.slider('MBA Percentage', 0, 100, 50)
    
    data = {
        'gender': gender,
        'ssc_p': ssc_p,
        'ssc_b': ssc_b,
        'hsc_p': hsc_p,
        'hsc_b': hsc_b,
        'hsc_s': hsc_s,
        'degree_p': degree_p,
        'degree_t': degree_t,
        'workex': workex,
        'etest_p': etest_p,
        'specialisation': specialisation,
        'mba_p': mba_p
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Align the input features with the training dataset
input_df_encoded = pd.get_dummies(input_df, columns=['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation'])
input_df_encoded = input_df_encoded.reindex(columns=data_encoded.columns.drop(['status', 'salary']), fill_value=0)

# Split the data
X = data_encoded.drop(columns=['status', 'salary'])
y = data_encoded['status']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
def rfc_model(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=14, criterion='entropy', random_state=41)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    acc_score = accuracy_score(y_test, prediction) * 100
    class_report = classification_report(y_test, prediction, output_dict=True)
    return model, acc_score, class_report

model, acc_score, class_report = rfc_model(x_train, y_train, x_test, y_test)

# Display model accuracy
st.subheader('Model Accuracy')
st.write(f'Accuracy: {acc_score:.2f}%')

# Make predictions
prediction = model.predict(input_df_encoded)

st.subheader('Prediction')
st.write(f'The predicted placement status is: {prediction[0]}')

st.subheader('User Input Parameters')
st.write(input_df)

st.subheader('Dataset')
st.write(data.head())
