
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# provide header name to browser
st.set_page_config(page_title='Iris project Aishwarya', layout='wide')

# add a title in body of browser
st.title('Iris project')

# Input fields
sepal_length = st.number_input('Sepal Length : ', min_value=0.00, step=0.01)
sepal_width = st.number_input('Sepal Width : ', min_value=0.00, step=0.01)
petal_length = st.number_input('Petal Length : ', min_value=0.00, step=0.01)
petal_width = st.number_input('Petal Width : ', min_value=0.00, step=0.01)

submit = st.button('predict')

st.subheader('Predictions Are : ')

def predict_species(scaler_path, model_path):
    with open(scaler_path, 'rb') as file1:
        scaler = pickle.load(file1)
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
        
    # Use feature names consistent with the model's training
    dct = {
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    }
    
    xnew = pd.DataFrame(dct)
    xnew_pre = scaler.transform(xnew)
    pred = model.predict(xnew_pre)
    probs = model.predict_proba(xnew_pre)
    max_prob = np.max(probs)
    return pred, max_prob

if submit:
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'
    pred, max_prob = predict_species(scaler_path, model_path)
    st.subheader(f'Predicted Species is : {pred[0]}')
    st.subheader(f'Probability of Prediction : {max_prob:.4f}')
    st.progress(max_prob)
