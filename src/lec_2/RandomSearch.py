from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.datasets.DataLoaderIris import X_train, X_test, y_train, y_test
import numpy as np
from typing import cast

# Parameter distributions
param_dist = {
    "n_estimators": np.arange(50, 200, 10),
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8]
}

# Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Train
random_search.fit(X_train, y_train)

# Best model
best_model = cast(RandomForestClassifier, random_search.best_estimator_)

# Prediction
y_pred = best_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    print("\n===== Random Search Results =====")
    print("Best Parameters:", random_search.best_params_)
    print(f"Best CV Accuracy: {random_search.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")