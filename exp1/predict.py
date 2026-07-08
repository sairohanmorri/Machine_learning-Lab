import pandas as pd

def predict_weight(model, height, age, gender):
    gender = 1 if gender.lower() == "male" else 0

    new_person = pd.DataFrame({
        "Height": [height],
        "Age": [age],
        "Gender": [gender]
    })

    prediction = model.predict(new_person)

    return prediction[0]