import joblib
from src.mood_labeler import detect_moods


class BookAnalyzer:
    """
    A class for analyzing book reviews using a trained sentiment model
    and a mood-labeling function.
    """

    def __init__(self, model_path="models/sentiment_model.joblib", vectorizer_path="models/tfidf_vectorizer.joblib"):
        """
        Load the trained sentiment model and TF-IDF vectorizer.
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict_sentiment(self, review_text):
        """
        Predict the sentiment of a review and return both label and confidence.
        """
        review_vec = self.vectorizer.transform([review_text])
        prediction = self.model.predict(review_vec)[0]

        probabilities = self.model.predict_proba(review_vec)[0]
        confidence = max(probabilities)

        return prediction, confidence

    def predict_moods(self, review_text):
        """
        Predict one or more mood labels from the review text.
        """
        return detect_moods(review_text)

    def analyze_review(self, review_text):
        """
        Analyze a review by returning sentiment, confidence, and mood labels.
        """
        sentiment, confidence = self.predict_sentiment(review_text)
        moods = self.predict_moods(review_text)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "moods": moods
        }
    
    
"""
if __name__ == "__main__":
    analyzer = BookAnalyzer()
    sample_review = "This book was moving, powerful, and deeply insightful."
    result = analyzer.analyze_review(sample_review)
    print(result)
"""