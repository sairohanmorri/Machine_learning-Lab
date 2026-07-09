from data import load_data
from preprocess import preprocess
from model import train_model
from predict import evaluate

data = load_data()

X_train,X_test,y_train,y_test = preprocess(data)

model = train_model(X_train,y_train)

evaluate(model,X_test,y_test)

print("\nPredict Admission")

gre=float(input("GRE Score : "))
toefl=float(input("TOEFL Score : "))
rating=float(input("University Rating : "))
sop=float(input("SOP Rating : "))
lor=float(input("LOR Rating : "))
cgpa=float(input("CGPA : "))
research=int(input("Research (0/1) : "))

prediction=model.predict([[gre,toefl,rating,sop,lor,cgpa,research]])

if prediction[0]==1:
    print("\nStudent is Likely to be Admitted")
else:
    print("\nStudent is Not Likely to be Admitted")