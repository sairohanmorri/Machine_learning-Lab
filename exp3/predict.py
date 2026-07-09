from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate(model,X_test,y_test):

    prediction=model.predict(X_test)

    print("\nAccuracy")
    print(accuracy_score(y_test,prediction))

    print("\nClassification Report")
    print(classification_report(y_test,prediction))

    cm=confusion_matrix(y_test,prediction)

    disp=ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.show()