import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plots(data,model,X_test,y_test):

    # Disease Distribution

    plt.figure(figsize=(6,4))

    sns.countplot(x="Outcome",data=data)

    plt.title("Disease Distribution")

    plt.show()

    # Age Distribution

    plt.figure(figsize=(6,4))

    plt.hist(data["Age"],bins=15)

    plt.title("Age Distribution")

    plt.xlabel("Age")

    plt.ylabel("Count")

    plt.show()

    # BMI vs Glucose

    plt.figure(figsize=(6,4))

    plt.scatter(data["BMI"],data["Glucose"])

    plt.xlabel("BMI")

    plt.ylabel("Glucose")

    plt.title("BMI vs Glucose")

    plt.show()

    # Correlation Heatmap

    plt.figure(figsize=(10,7))

    sns.heatmap(data.corr(),annot=True,cmap="coolwarm")

    plt.title("Correlation Heatmap")

    plt.show()

    # Confusion Matrix

    prediction=model.predict(X_test)

    cm=confusion_matrix(y_test,prediction)

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues')

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()