import pandas as pd

def load_data():
    data = {
        'Weight': [79,69,73,95,82,55,69,71,64,69],
        'Height': [1.80,1.68,1.82,1.70,1.87,1.55,1.50,1.78,1.67,1.64],
        'Age': [35,39,25,60,27,18,89,42,16,52],
        'Gender': [
            'Male','Male','Male','Male','Male',
            'Female','Female','Female','Female','Female'
        ]
    }

    return pd.DataFrame(data)