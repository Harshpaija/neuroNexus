import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path='task-1/tested.csv'):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {os.getcwd()}")
        return None
    df = pd.read_csv(file_path)
    if 'Survived' not in df.columns:
        print("Error: 'Survived' column is missing in the dataset.")
        return None
    return df

def preprocess_data(df):
    df = df.copy()
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("=============== Model Evaluation ===============")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def feature_importance(model, feature_names):
    coef = pd.Series(model.coef_[0], index=feature_names)
    coef_sorted = coef.sort_values()
    print("\nTop Influential Features:")
    print(coef_sorted.tail(5))
    print("\nLeast Influential Features:")
    print(coef_sorted.head(5))

    # Optional: Plot feature importance
    plt.figure(figsize=(10, 6))
    coef_sorted.plot(kind='barh', title="Feature Importance")
    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    if df is None:
        return

    df = preprocess_data(df)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    print(f"Dataset size: {len(df)} rows")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    feature_importance(model, X.columns)

if __name__ == "__main__":
    main()