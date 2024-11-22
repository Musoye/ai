import pandas as pd
import numpy as np
import pickle

def load_model():
    with open('survivor.pkl', 'rb') as file:
        n_model = pickle.load(file)
    return n_model

def predict_survivor(gender, age, sbps, fare):
    n_model = load_model()
    loaded = n_model["model"]
    X = np.array([gender, age, sbps, fare])
    return loaded.predict(X.reshape(1, -1))[0]
