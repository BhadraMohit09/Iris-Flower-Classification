# iris_classification.py

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load the dataset
def load_data():
    # Option 1: Using Scikit-learn’s built-in dataset
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: species
    return X, y

    # Uncomment this for loading from CSV
    # df = pd.read_csv('path_to_your_iris_dataset.csv')
    # X = df.drop('species', axis=1)  # Replace 'species' with the actual label column name if different
    # y = df['species']
    # return X, y

# Split the dataset
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main function
def main():
    X, y = load_data()  # Load data
    X_train, X_test, y_train, y_test = split_data(X, y)  # Split data
    model = train_model(X_train, y_train)  # Train model
    evaluate_model(model, X_test, y_test)  # Evaluate model

# Execute the main function
if __name__ == "__main__":
    main()
