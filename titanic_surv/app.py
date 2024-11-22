import streamlit as st
from model_load import predict_survivor

st.write("Predict Survival on the Titanic")

gender_options = ['male', 'female']

gender = st.selectbox('Which Gender??', gender_options)

age = st.number_input('How old are you?', min_value=0, max_value=100, value=20)

sbps = st.number_input('How many siblings or spouses are you traveling with?', min_value=0, max_value=10, value=0)

fare = st.number_input('How much Fare?', )

clicked = st.button('Predict Survival')

if clicked:
    gender = 1 if gender == 'male' else 0
    prediction = predict_survivor(int(gender), float(age), float(sbps), float(fare))
    if prediction == 1:
        st.write('Survived')
    else:
        st.write('Did not survive')