from sklearn.datasets import load_breast_cancer         # Breast Cancer Dataset
from sklearn.model_selection import train_test_split    # train test split
import pandas as pd                                     

# Load the breast cancer dataset
data = load_breast_cancer()  

# Assign features and target variable
X = data.data # type: ignore
y = data.target # type: ignore

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It is for func Test if it use as module then it will not run the code below just comment it out
if __name__ == "__main__":
    print(f"Feature Names: {data.feature_names}") # type: ignore
    print(f"Target Names: {data.target_names}") # type: ignore