import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Data loading and preprocessing
def load_and_preprocess():
    # Load dataset
    file_path = 'imputed_data1.csv'  # Using dataset without missing values
    data = pd.read_csv(file_path)

    # Select required features
    required_features = [
        'Gender', 'Sleep_night', 'IADL_score', 'Loneliness',
        'Medical Insurance', 'BMI', 'Digestive', 'Martial_status', 'Heart'
    ]

    # Target variable
    target_col = 'Depression'

    # Features and target
    X = data[required_features]
    y = data[target_col]

    return X, y

# Load data and train model
X, y = load_and_preprocess()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Create Streamlit interface
st.title('Depression Risk Predictor')

# User input fields
gender = st.selectbox('Gender (0=Male, 1=Female)', [0, 1])
sleep_night = st.number_input('Hours of sleep per night', min_value=0, max_value=24, value=7)
IADL_score = st.number_input('IADL Score', min_value=0, max_value=100, value=50)
loneliness = st.selectbox('Loneliness level (1=Rarely, 2=Sometimes, 3=Occasionally, 4=Often)', [1, 2, 3, 4])
medical_insurance = st.selectbox('Medical Insurance (0=No, 1=Yes)', [0, 1])
BMI = st.number_input('BMI Value', min_value=10.0, max_value=50.0, value=25.0)
digestive = st.selectbox('Digestive disorders (0=No, 1=Yes)', [0, 1])
martial_status = st.selectbox('Marital Status (1=Married, 2=Single, 3=Divorced/Widowed/Separated)', [1, 2, 3])
heart = st.selectbox('Heart disease (0=No, 1=Yes)', [0, 1])

# User input data
user_input = np.array(
    [[gender, sleep_night, IADL_score, loneliness, medical_insurance, BMI, digestive, martial_status, heart]])

# Prediction button
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(user_input)
    if prediction == 1:
        st.write("High risk of depression")
    else:
        st.write("Low risk of depression")