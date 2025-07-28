import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load the MNIST dataset (CSV format)
# Replace 'mnist.csv' with your actual file path if different
print("Loading data...")
df = pd.read_csv('mnist_train.csv')

# 2. Split features and labels
X = df.iloc[:, 1:]  # pixel values
y = df.iloc[:, 0]   # labels

# 3. Normalize the data
print("Normalizing pixel values...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split into train and test sets (80% train, 20% test)
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Initialize multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC()
}

best_accuracy = 0
best_model = None
best_model_name = ""

# 6. Train and evaluate each model
print("Training and evaluating models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# 7. Show classification report and confusion matrix for best model
print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")
y_best_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_best_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_best_pred))

# 8. Save the best model
joblib.dump(best_model, 'mnist_best_model.pkl')
print(f"\nSaved best model as 'mnist_best_model.pkl'")
