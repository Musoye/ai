import streamlit as st
from load_model import predict_diabetes_patient

st.write("Predict Diabetic Chance")


pregnancy = st.number_input('How old is your pregnacy?', min_value=0, max_value=15, value=1)

glucose = st.number_input('What\'s your glucose level?', min_value=0, max_value=200, value=10)

bldpr = st.number_input('What\'s your blood pressure level?', min_value=0, max_value=200, value=10)

skinthick = st.number_input('What\'s your skin thickness?', min_value=0, max_value=200, value=10)

insulin = st.number_input('What\'s your insulin level?', min_value=0, max_value=200, value=10)

bmi = st.number_input('What\'s your BMI?', min_value=0.0, max_value=50.0, value=10.0)

diapefunc = st.number_input('What\'s your Diabetes pedigree function?', min_value=0.0, max_value=10.0, value=1.0)

age = st.number_input('What\'s your age?', min_value=0, max_value=150, value=10)

clicked = st.button('Predict Diabetic Chance')

if clicked:
    prediction = predict_diabetes_patient(pregnancy, glucose, bldpr, skinthick, insulin, bmi, diapefunc, age)
    if prediction == 1:
        st.write('Diabetes Patient')
    else:
        st.write('Not a Diabetes Patient')