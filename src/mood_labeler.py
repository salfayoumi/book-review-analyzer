def detect_moods(text):
    """
    Detect one or more mood labels from a review text
    using keyword-based matching.
    """
    text = str(text).lower()

    mood_keywords = {
        "uplifting": ["inspiring", "hopeful", "motivating", "heartwarming", "feel-good"],
        "dark": ["disturbing", "bleak", "unsettling", "grim", "haunting"],
        "funny": ["hilarious", "witty", "laugh-out-loud", "amusing", "clever"],
        "emotional": ["moving", "tearful", "powerful", "touching", "devastating"],
        "thought-provoking": ["insightful", "challenging", "eye-opening", "deep", "reflective"],
    }

    labels = []

    for mood, keywords in mood_keywords.items():
        if any(keyword in text for keyword in keywords):
            labels.append(mood)

    if not labels:
        labels.append("thought-provoking")

    return labels

"""
if __name__ == "__main__":
    sample_text = "This book was inspiring, heartwarming, and very deep."
    print(detect_moods(sample_text))
"""