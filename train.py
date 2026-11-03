from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import json

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save trained model
joblib.dump(model, "iris_model.pkl")

# Save evaluation results
results = {
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist(),
    "class_names": iris.target_names.tolist()
}

with open("evaluation.json", "w") as file:
    json.dump(results, file, indent=4)

print("Model trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Evaluation results saved successfully!")