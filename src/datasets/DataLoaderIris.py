from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()

# Assign features and target variable
X = data.data # type: ignore
y = data.target # type: ignore

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

if __name__ == "__main__":
    print(f"\n\nFeature Names: {data.feature_names}")  # type: ignore
    print(f"Target Names: {data.target_names}") # type: ignore