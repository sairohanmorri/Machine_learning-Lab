import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Salary_Data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("X_train")
print(X_train)

print("X_test")
print(X_test)

print("y_train")
print(y_train)

print("y_test")
print(y_test)