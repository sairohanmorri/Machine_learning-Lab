from data import load_data
from preprocess import preprocess
from model import train_model
from predict import predict_weight

# Load data
df = load_data()

# Preprocess data
X, y = preprocess(df)

# Train model
model = train_model(X, y)

# Print equation information
print("Intercept:")
print(model.intercept_)

print("\nCoefficients")

print(f"Height : {model.coef_[0]}")
print(f"Age    : {model.coef_[1]}")
print(f"Gender : {model.coef_[2]}")

# Model Accuracy
print("\nR² Score:")
print(model.score(X, y))

# Prediction Example
weight = predict_weight(model, 1.75, 30, "Male")

print("\nPrediction")
print(f"Predicted Weight: {weight:.2f} kg")