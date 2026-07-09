from data import load_data
from preprocess import preprocess
from model import train_model
from predict import evaluate
from visualize import plots

data=load_data()

X_train,X_test,y_train,y_test=preprocess(data)

model=train_model(X_train,y_train)

evaluate(model,X_test,y_test)

plots(data,model,X_test,y_test)

print("\nPredict Disease")

preg=float(input("Pregnancies : "))

glucose=float(input("Glucose : "))

bp=float(input("Blood Pressure : "))

skin=float(input("Skin Thickness : "))

insulin=float(input("Insulin : "))

bmi=float(input("BMI : "))

dpf=float(input("Diabetes Pedigree Function : "))

age=float(input("Age : "))

prediction=model.predict([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])

if prediction[0]==1:

    print("\nPerson is Susceptible to Diabetes")

else:

    print("\nPerson is Not Susceptible to Diabetes")