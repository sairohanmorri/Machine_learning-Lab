from sklearn.model_selection import train_test_split

def preprocess(data):

    print("\nChecking Null Values\n")

    print(data.isnull().sum())

    data = data.dropna()

    # Convert Chance of Admit into Binary Classes
    data["Admit"] = (data["Chance of Admit "] >= 0.75).astype(int)

    X = data[[
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR ",
        "CGPA",
        "Research"
    ]]

    y = data["Admit"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train,X_test,y_train,y_test