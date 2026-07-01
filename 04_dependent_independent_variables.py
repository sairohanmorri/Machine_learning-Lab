import pandas as pd

data = pd.read_csv("Salary_Data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

print("Independent Variable (X):")
print(X)

print("Dependent Variable (y):")
print(y)