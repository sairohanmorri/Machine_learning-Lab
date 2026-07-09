from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def evaluate(model,X_test,y_test):

    prediction=model.predict(X_test)

    print("\nAccuracy")

    print(accuracy_score(y_test,prediction))

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test,prediction))

    print("\nClassification Report")

    print(classification_report(y_test,prediction))