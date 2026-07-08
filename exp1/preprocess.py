def preprocess(df):
    df["Gender"] = df["Gender"].map({
        "Male": 1,
        "Female": 0
    })

    X = df[["Height", "Age", "Gender"]]
    y = df["Weight"]

    return X, y