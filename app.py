import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the trained model and scaler
model = pickle.load(open('heart_failure_prediction.pkl', 'rb'))  # Load the logistic regression model
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler

#streamlit page
st.title('Heart Failure Prediction')
st.header('Enter the following details to predict heart failure:')
age = st.number_input('Age', min_value=0, max_value=120, value=25)
anaemia = st.selectbox('Anaemia', [0, 1])
creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase', min_value=0, max_value=10000, value=100)
diabetes = st.selectbox('Diabetes', [0, 1])
ejection_fraction = st.number_input('Ejection Fraction', min_value=0, max_value=100, value=50)
high_blood_pressure = st.selectbox('High Blood Pressure', [0, 1])
platelets = st.number_input('Platelets', min_value=0, max_value=1000000, value=10000)
serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0, max_value=10.0, value=1.0)
serum_sodium = st.number_input('Serum Sodium', min_value=0, max_value=200, value=100)
sex = st.selectbox('Sex', ('Male', 'Female'))

# Map 'Male' to 1 and 'Female' to 0
if sex == 'Male':
    sex_value = 1
else:
    sex_value = 0

smoking = st.selectbox('Smoking', [0, 1])
time = st.number_input('Time', min_value=0, max_value=1000, value=100)

input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex_value, smoking, time]])

if st.button('Predict'):
  scaled_input = scaler.transform(input_data)
  prediction = model.predict(scaled_input)
  if prediction[0] == 0:
    st.write('The person is not at risk of heart failure.')
  else:
    st.write('The person is at risk of heart failure.')

