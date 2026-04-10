from src.analyzer import BookAnalyzer

analyzer = BookAnalyzer()

samples = [
    "This book was inspiring and heartwarming.",
    "A very disturbing and dark story.",
    "Absolutely hilarious and fun to read.",
    "Deep and insightful, made me think a lot.",
    "Very emotional and touching narrative.",
    "Not very interesting, a bit boring.",
    "Powerful and moving experience.",
    "Funny at times but also thoughtful.",
    "Quite unsettling and grim atmosphere.",
    "A motivating and uplifting story."
]

for text in samples:
    result = analyzer.analyze_review(text)
    print(f"\nText: {text}")
    print(result)