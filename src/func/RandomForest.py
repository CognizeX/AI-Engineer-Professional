from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from DataLoader import X_train, y_train, X_test, y_test

# Train a Random Forest Classifier with default parameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

if __name__ == "__main__":
    # Predict and evaluate the model
    y_pred = rf_default.predict(X_test)
    default_accuracy = accuracy_score(y_test, y_pred)
    default_confusion_matrix = confusion_matrix(y_test, y_pred)
    default_classification_report = classification_report(y_test, y_pred)
    
    # See the results
    print(f"\n[Default] Accuracy: {default_accuracy:4f}\n")
    print(f"[Default] Confusion Matrix:\n{default_confusion_matrix}\n")
    print(f"[Default] Classification Report:\n{default_classification_report}")