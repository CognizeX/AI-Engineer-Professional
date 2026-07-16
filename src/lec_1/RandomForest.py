from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.datasets.DataLoaderIris import X_train, X_test, y_train, y_test

# Random Forest Classifier with default parameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

# Random Forest Classifier with adjusted hyperparameters
rf_tuned = RandomForestClassifier(
    #n_estimators=200, 
    n_estimators=400, # increase by 2x
    #n_estimators=800, # increase by 4x
    # -------------------------------------
    #max_depth=10, 
    #max_depth=20,  # increase by 2x
    max_depth=30,  # increase by 3x
    # -------------------------------------
    #min_samples_split=5, 
    #min_samples_split=10,  # increase by 2x
    #min_samples_split=20,  # increase by 4x
    min_samples_split=30,  # increase by 6x
    # -------------------------------------
    #min_samples_leaf=1,
    #min_samples_leaf=2,  # increase by 2x
    min_samples_leaf=4,  # increase by 4x
    # -------------------------------------
    random_state=42
)
rf_tuned.fit(X_train, y_train)

if __name__ == "__main__":
    try:
        select_opt = int(input("\nSelect an option:\n1. Default Random Forest\n2. Tuned Random Forest\nEnter your choice (1 or 2): "))
    except ValueError:
        print("Invalid input. Please enter 1 or 2.")
        exit()
        
    if select_opt not in [1, 2]:
        print("Invalid choice. Please enter 1 or 2.")
        exit()
        
    elif select_opt == 1:
            rf_model = rf_default
            model_name = "Default Random Forest"
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            default_confusion_matrix = confusion_matrix(y_test, y_pred)
            default_classification_report = classification_report(y_test, y_pred)
            
            print(f"\n{model_name} Accuracy: {accuracy:4f}\n")
            print(f"{model_name} Confusion Matrix:\n{default_confusion_matrix}\n")
            print(f"{model_name} Classification Report:\n{default_classification_report}")
            
            
    elif select_opt == 2:
            rf_model = rf_tuned
            model_name = "Tuned Random Forest"
            y_pred_tuned = rf_tuned.predict(X_test)
            accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
            confusion_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)
            classification_report_tuned = classification_report(y_test, y_pred_tuned)
    
            print(f"\n[Tuned] Accuracy: {accuracy_tuned:4f}\n")
            print(f"[Tuned] Confusion Matrix:\n{confusion_matrix_tuned}\n")
            print(f"[Tuned] Classification Report:\n{classification_report_tuned}")
    else:
        print("Invalid choice. Please enter 1 or 2.")


    
