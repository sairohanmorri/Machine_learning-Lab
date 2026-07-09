import pandas as pd

def load_data():
    data = pd.read_csv("taxi.csv")
    return data