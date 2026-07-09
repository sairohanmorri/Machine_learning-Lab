from data import load_data
from preprocess import preprocess
from model import train_model
from predict import evaluate

data = load_data()

X_train, X_test, y_train, y_test = preprocess(data)

model = train_model(X_train, y_train)

evaluate(model, X_test, y_test)