from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.datasets.DataLoaderIris import X_train, X_test, y_train, y_test

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initialize the Grid Search 
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Train the model using Grid Search
grid_search.fit(X_train, y_train)

# Get the best model
best_grid_model = grid_search.best_estimator_

# Predict on the test set
y_predict_grid = best_grid_model.predict(X_test)

# Calculate accuracy
accuracy_grid = accuracy_score(y_test, y_predict_grid)

if __name__ == "__main__":
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    print(f"Test Set Accuracy: {accuracy_grid:.4f}")