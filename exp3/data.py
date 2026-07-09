import pandas as pd

def load_data():

    data = pd.read_csv("admission.csv")

    print("\nDataset Loaded Successfully\n")

    print(data.head())

    return data