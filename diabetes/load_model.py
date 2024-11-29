import pandas as pd
import numpy as np
import pickle

def load_model():
    with open('dia_training.pkl', 'rb') as file:
        n_model = pickle.load(file)
    return n_model

def predict_diabetes_patient(pregnancy, glucose, bldpr, skinthick, insulin, bmi, diapefunc, age):
    n_model = load_model()
    loaded = n_model["model"]
    X = np.array([pregnancy, glucose, bldpr, skinthick, insulin, bmi, diapefunc, age])
    return loaded.predict(X.reshape(1, -1))[0]