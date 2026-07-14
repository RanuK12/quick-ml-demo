# iris_demo.py
"""Simple Iris dataset logistic regression demo.
Loads the Iris dataset, trains a LogisticRegression model, and provides a predict function.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data
iris = load_iris()
X = iris.data  # shape (150, 4)
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Mapping target to species name
species = iris.target_names

def predict(sepal_length, sepal_width, petal_length, petal_width):
    """Predict Iris species given measurements.
    Returns the species name as string.
    """
    arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(arr)[0]
    return species[pred]

if __name__ == "__main__":
    # Example usage
    example = [5.1, 3.5, 1.4, 0.2]
    print(f"Predicted species: {predict(*example)}")
