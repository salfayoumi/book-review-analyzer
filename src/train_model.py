import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


print("train_model.py started")

# Convert rating → sentiment
def label_sentiment(score):
    if score >= 4:
        return "positive"
    elif score == 3:
        return "neutral"
    else:
        return "negative"


def main():
    # Load cleaned dataset
    print("Loading dataset...")
    df = pd.read_csv("data/sample_books.csv")
    print("Dataset loaded:", df.shape)

    # Create sentiment column
    df["sentiment"] = df["rating"].apply(label_sentiment)

    # Features and labels
    X = df["review"]
    y = df["sentiment"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    # Predictions
    y_pred = model.predict(X_test_vec)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score (weighted):", f1_score(y_test, y_pred, average="weighted"))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/sentiment_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()