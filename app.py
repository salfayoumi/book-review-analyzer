import pandas as pd
import streamlit as st
import plotly.express as px

from src.analyzer import BookAnalyzer
from src.mood_labeler import detect_moods


st.set_page_config(
    page_title="Book Review Analyzer",
    page_icon="📚",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "Home"


@st.cache_data
def load_data():
    """
    Load the processed dataset used by the application.
    """
    return pd.read_csv("data/sample_books.csv")


@st.cache_data
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns used in the application such as review length,
    mood labels, and time-based features if available.
    """
    prepared_df = df.copy()

    prepared_df["review_length"] = prepared_df["review"].astype(str).apply(len)
    prepared_df["moods"] = prepared_df["review"].apply(detect_moods)
    prepared_df["mood_str"] = prepared_df["moods"].apply(lambda x: ", ".join(x))

    if "review_time" in prepared_df.columns:
        prepared_df["review_time"] = pd.to_numeric(
            prepared_df["review_time"], errors="coerce"
        )
        prepared_df["review_datetime"] = pd.to_datetime(
            prepared_df["review_time"], unit="s", errors="coerce"
        )
        prepared_df["review_year"] = prepared_df["review_datetime"].dt.year

    return prepared_df


def go_to(page_name: str):
    """
    Update the current page in session state.
    """
    st.session_state.page = page_name


def show_back_button():
    """
    Display a button that returns the user to the Home page.
    """
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Home"):
            go_to("Home")
            st.rerun()


def show_home(df: pd.DataFrame):
    """
    Display the welcome section of the application.
    """
    st.title("📚 Book Review Analyzer & Mood-Based Recommender")

    st.markdown(
        """
        Welcome to the **Book Review Analyzer & Mood-Based Recommender**.

        This application analyzes book reviews using a complete data science workflow. It combines:

        - **Sentiment Analysis** to classify reviews as positive, neutral, or negative  
        - **Mood Labeling** to detect moods such as uplifting, dark, funny, emotional, and thought-provoking  
        - **Recommendation Logic** to suggest books based on selected moods  
        - **Interactive Data Insights** to help users explore the dataset visually  
        """
    )

    st.info(
        "Use the buttons below or the navigation menu on the left to explore the application."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎭 Mood-Based Recommendation")
        st.write(
            "Choose a mood and discover the top 5 books whose reviews match that emotional tone."
        )
        if st.button("Go to Mood-Based Recommendation"):
            go_to("Mood-Based Recommendation")
            st.rerun()

    with col2:
        st.subheader("🤖 Review Analysis")
        st.write(
            "Enter a custom review and receive sentiment prediction, confidence score, and mood labels."
        )
        if st.button("Go to Review Analysis"):
            go_to("Review Analysis")
            st.rerun()

    st.markdown("---")

    st.subheader("📊 Explore the Dataset")
    st.write(
        "Inspect charts and trends from the review dataset in the Data Insights section."
    )
    if st.button("Go to Data Insights"):
        go_to("Data Insights")
        st.rerun()

    st.markdown("---")

    st.subheader("📌 Quick Overview")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Dataset Size", f"{len(df):,}")

    with col4:
        st.metric("Unique Books", f"{df['Title'].nunique():,}")

    with col5:
        st.metric("Mood Categories", "5")

    st.markdown("---")

    st.subheader("ℹ️ About the App")
    st.write(
        "This tool is designed to make review analysis easier and more interactive. "
        "It can help users explore review sentiment, identify emotional tone, and discover books based on selected moods."
    )


def show_recommendations(df: pd.DataFrame):
    """
    Display mood-based book recommendations.
    """
    show_back_button()

    st.title("🎭 Mood-Based Recommendation")
    st.write(
        "Select a mood below to get the top 5 recommended books based on review sentiment patterns and popularity."
    )

    selected_mood = st.selectbox(
        "Select a mood:",
        ["uplifting", "dark", "funny", "emotional", "thought-provoking"]
    )

    filtered = df[df["moods"].apply(lambda moods: selected_mood in moods)]

    if filtered.empty:
        st.warning("No books were found for the selected mood.")
        return

    recommendations = (
        filtered.groupby("Title")
        .agg(
            avg_rating=("rating", "mean"),
            review_count=("review", "count"),
            example_mood=("mood_str", "first")
        )
        .reset_index()
    )

    recommendations["normalized_review_count"] = (
        recommendations["review_count"] / recommendations["review_count"].max()
    )

    recommendations["final_score"] = (
        recommendations["avg_rating"] * 0.7
        + recommendations["normalized_review_count"] * 0.3
    )

    recommendations = (
        recommendations.sort_values(by="final_score", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    recommendations.index = recommendations.index + 1
    recommendations.index.name = "Rank"

    st.subheader("Top 5 Recommended Books")

    st.dataframe(
        recommendations[
            ["Title", "avg_rating", "review_count", "example_mood", "final_score"]
        ].rename(
            columns={
                "Title": "Book Title",
                "avg_rating": "Average Rating",
                "review_count": "Review Count",
                "example_mood": "Mood Labels",
                "final_score": "Final Score",
            }
        ),
        use_container_width=True
    )

    st.caption(
        "Recommendations are ranked using a combination of average rating and review popularity."
    )


def show_review_analysis(analyzer: BookAnalyzer):
    """
    Display review sentiment and mood analysis.
    """
    show_back_button()

    st.title("🤖 Review Analysis")
    st.write(
        "Enter a review text below to analyze its sentiment and mood labels."
    )

    user_review = st.text_area(
        "Enter a review text:",
        height=180,
        placeholder="Example: This book was moving, powerful, and deeply insightful."
    )

    if st.button("Analyze Review"):
        if user_review.strip():
            result = analyzer.analyze_review(user_review)

            st.subheader("Analysis Result")
            st.success("Review analyzed successfully.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", result["sentiment"].capitalize())

            with col2:
                st.metric("Confidence", f"{result['confidence']:.2f}")

            with col3:
                st.metric("Mood Count", len(result["moods"]))

            st.write("**Mood Labels:**", ", ".join(result["moods"]))

            if result["confidence"] < 0.60:
                st.warning(
                    "This prediction has relatively low confidence, so the result may be uncertain."
                )
        else:
            st.warning("Please enter some text first.")


def show_data_insights(df: pd.DataFrame):
    """
    Display visual insights from the dataset using interactive Plotly charts.
    """
    show_back_button()

    st.title("📊 Data Insights")
    st.write("Explore visual patterns in the sampled review dataset.")

    chart_option = st.selectbox(
        "Choose a chart:",
        [
            "Rating Distribution",
            "Review Length by Rating",
            "Top Most Reviewed Books",
            "Average Rating Over Time",
            "Review Count Over Time"
        ]
    )

    if chart_option == "Rating Distribution":
        st.subheader("Rating Distribution")

        min_rating, max_rating = st.slider(
            "Filter rating range:",
            min_value=1,
            max_value=5,
            value=(1, 5)
        )

        filtered_df = df[(df["rating"] >= min_rating) & (df["rating"] <= max_rating)]
        counts = (
            filtered_df["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        counts.columns = ["Rating", "Count"]

        fig = px.bar(
            counts,
            x="Rating",
            y="Count",
            text="Count",
            title="Rating Distribution"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Count",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.write(
            "This chart shows the distribution of review scores within the selected rating range."
        )

    elif chart_option == "Review Length by Rating":
        st.subheader("Review Length by Rating")

        sample_size = st.slider(
            "Number of points to display:",
            min_value=500,
            max_value=min(5000, len(df)),
            value=min(2000, len(df)),
            step=500
        )

        sampled_df = df.sample(sample_size, random_state=42)

        fig = px.scatter(
            sampled_df,
            x="rating",
            y="review_length",
            opacity=0.45,
            title="Review Length vs Rating"
        )
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Review Length",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.write(
            "This chart shows how review lengths vary across different rating values."
        )

    elif chart_option == "Top Most Reviewed Books":
        st.subheader("Top Most Reviewed Books")

        top_n = st.slider(
            "Number of books to display:",
            min_value=5,
            max_value=15,
            value=10
        )

        top_books = df["Title"].value_counts().head(top_n).reset_index()
        top_books.columns = ["Book Title", "Review Count"]

        fig = px.bar(
            top_books,
            x="Review Count",
            y="Book Title",
            orientation="h",
            text="Review Count",
            title=f"Top {top_n} Most Reviewed Books"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            template="plotly_white",
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.write(
            "This chart highlights the books that appear most frequently in the sampled review dataset."
        )

    elif chart_option == "Average Rating Over Time":
        st.subheader("Average Rating Over Time")

        if "review_year" not in df.columns:
            st.warning(
                "This chart is unavailable because the processed dataset does not currently include review time information."
            )
        else:
            yearly_rating = (
                df.dropna(subset=["review_year"])
                .groupby("review_year", as_index=False)["rating"]
                .mean()
            )
            yearly_rating.columns = ["Year", "Average Rating"]

            fig = px.line(
                yearly_rating,
                x="Year",
                y="Average Rating",
                markers=True,
                title="Average Review Score Over Time"
            )
            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            st.write(
                "This chart shows how average review scores change over time."
            )

    elif chart_option == "Review Count Over Time":
        st.subheader("Review Count Over Time")

        if "review_year" not in df.columns:
            st.warning(
                "This chart is unavailable because the processed dataset does not currently include review time information."
            )
        else:
            yearly_count = (
                df.dropna(subset=["review_year"])
                .groupby("review_year")
                .size()
                .reset_index(name="Review Count")
            )
            yearly_count.columns = ["Year", "Review Count"]

            fig = px.line(
                yearly_count,
                x="Year",
                y="Review Count",
                markers=True,
                title="Number of Reviews Over Time"
            )
            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            st.write(
                "This chart shows how the number of reviews changes over time in the sampled dataset."
            )


def main():
    """
    Run the Streamlit application.
    """
    df = load_data()
    df = prepare_data(df)
    analyzer = BookAnalyzer()

    pages = ["Home", "Mood-Based Recommendation", "Review Analysis", "Data Insights"]

    st.sidebar.title("Navigation")
    sidebar_page = st.sidebar.radio(
        "Go to:",
        pages,
        index=pages.index(st.session_state.page)
    )
    st.session_state.page = sidebar_page

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application analyzes book reviews, predicts sentiment, detects moods, and recommends books."
    )

    if st.session_state.page == "Home":
        show_home(df)
    elif st.session_state.page == "Mood-Based Recommendation":
        show_recommendations(df)
    elif st.session_state.page == "Review Analysis":
        show_review_analysis(analyzer)
    elif st.session_state.page == "Data Insights":
        show_data_insights(df)


if __name__ == "__main__":
    main()