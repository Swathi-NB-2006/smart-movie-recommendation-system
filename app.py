"""
Smart Movie Recommendation System
A beginner-friendly Streamlit app using TF-IDF and cosine similarity.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import DatasetError, dataset_overview, load_dataset
from recommendation_engine import MOOD_PROFILES, RecommendationEngine

PAGES = [
    "🏠 Home",
    "🔍 Discover",
    "😊 Mood",
    "⭐ Personalized",
    "🏆 Top Rated",
    "🎭 Genres",
    "🎞️ Watchlist",
    "ℹ️ About",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #2a2a30 0%, #121214 42%);
            color: #f0ece4;
        }
        [data-testid="stSidebar"] {
            background: #0f0f12;
            border-right: 1px solid #3a3428;
        }
        [data-testid="stSidebar"] * {
            color: #f0ece4 !important;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #f5e6c8;
            letter-spacing: 0.3px;
            margin-bottom: 0.2rem;
        }
        .hero-sub {
            color: #cbbd9a;
            font-size: 1.05rem;
            margin-bottom: 1.4rem;
        }
        .gold-rule {
            height: 2px;
            background: linear-gradient(90deg, #d4af37, transparent);
            margin: 0.4rem 0 1.2rem 0;
        }
        .metric-card, .movie-card, .panel {
            background: #1e1e22;
            border: 1px solid #3a3428;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }
        .movie-title {
            color: #f5e6c8;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .muted {
            color: #b7b1a6;
            font-size: 0.92rem;
        }
        .reason {
            color: #d4af37;
            font-size: 0.9rem;
            margin-top: 0.55rem;
        }
        .chip {
            display: inline-block;
            background: #2b2b31;
            color: #d4af37;
            border: 1px solid #d4af37;
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            font-size: 0.8rem;
        }
        .stButton > button, button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-primary"], div.stButton button {
            background: #d4af37 !important;
            color: #161618 !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover, button[data-testid="stBaseButton-secondary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
            background: #e6c65a !important;
            color: #161618 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #d4af37;
        }
        h1, h2, h3 {
            color: #f5e6c8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Preparing movie data and TF-IDF model...")
def get_system():
    data = load_dataset()
    engine = RecommendationEngine(data["movies"])
    overview = dataset_overview(data["movies"], data["ratings"])
    return data, engine, overview


def init_state() -> None:
    st.session_state.setdefault("watchlist", [])
    st.session_state.setdefault("selected_movie", "Toy Story (1995)")
    st.session_state.setdefault("nlp_query", "")
    st.session_state.setdefault("last_recommendations", [])
    st.session_state.setdefault("last_source_title", "")


def rating_label(value) -> str:
    if value is None:
        return "No ratings yet"
    return f"{value:.2f} / 5"


def add_to_watchlist(movie: dict) -> None:
    existing = {item["movieId"] for item in st.session_state.watchlist}
    if movie["movieId"] not in existing:
        st.session_state.watchlist.append(
            {
                "movieId": movie["movieId"],
                "title": movie["title"],
                "genres": movie["genres"],
                "avg_rating": movie["avg_rating"],
            }
        )


def render_movie_cards(movies: list[dict], key_prefix: str, show_similarity: bool = True) -> None:
    if not movies:
        st.info("No recommendations found. Try a different movie, mood, or search phrase.")
        return

    for index, movie in enumerate(movies):
        with st.container():
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="muted">{movie["genres"]}</div>', unsafe_allow_html=True)
                details = [f"Average rating: {rating_label(movie['avg_rating'])}"]
                details.append(f"Ratings: {movie['rating_count']}")
                if show_similarity and movie.get("similarity") is not None:
                    details.append(f"Similarity: {movie['similarity']}%")
                st.markdown(f'<div class="muted">{" • ".join(details)}</div>', unsafe_allow_html=True)
                if movie.get("because"):
                    st.markdown(f'<div class="muted">Because you liked {movie["because"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="reason">Why this movie? {movie["reason"]}</div>', unsafe_allow_html=True)
            with right:
                if st.button("Add", key=f"{key_prefix}-add-{movie['movieId']}-{index}"):
                    add_to_watchlist(movie)
                    st.success(f"Added {movie['title']} to watchlist.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")


def page_home(engine: RecommendationEngine, overview: dict) -> None:
    st.markdown('<div class="hero-title">Smart Movie Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Discover movies using machine learning, NLP and personalized content analysis.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    st.write(
        "This project recommends movies by comparing **content**, not by chatting with a generative AI model. "
        "It reads titles, genres and user tags, turns them into TF-IDF vectors, and ranks similar films with cosine similarity."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{overview['movie_count']:,}")
    c2.metric("Ratings", f"{overview['rating_count']:,}")
    c3.metric("Users", f"{overview['user_count']:,}")
    c4.metric("Average rating", f"{overview['average_rating']:.2f}")

    st.subheader("Try these first")
    cols = st.columns(4)
    if cols[0].button("Find films like Toy Story"):
        st.session_state.selected_movie = "Toy Story (1995)"
        st.session_state.nav = "🔍 Discover"
        st.rerun()
    if cols[1].button("Search: funny adventure movie"):
        st.session_state.nlp_query = "funny adventure movie"
        st.session_state.nlp_box = "funny adventure movie"
        st.session_state.nav = "🔍 Discover"
        st.rerun()
    if cols[2].button("Open Romantic mood"):
        st.session_state.chosen_mood = "Romantic"
        st.session_state.nav = "😊 Mood"
        st.rerun()
    if cols[3].button("Search: intelligent thriller"):
        st.session_state.nlp_query = "intelligent thriller"
        st.session_state.nlp_box = "intelligent thriller"
        st.session_state.nav = "🔍 Discover"
        st.rerun()

    st.subheader("How a recommendation is made")
    st.markdown(
        """
        1. Movie data is loaded from MovieLens (`movies.csv`, `ratings.csv`, `tags.csv`).
        2. A content string is built from **title + genres + tags**.
        3. TF-IDF converts that text into numbers.
        4. Cosine similarity ranks the closest movies.
        5. Ratings are shown as extra information. They are **not** part of the TF-IDF model.
        """
    )

    genre_counts = (
        engine.movies["genre_list"].explode().dropna().value_counts().head(10).reset_index()
    )
    genre_counts.columns = ["Genre", "Movies"]
    chart = px.bar(
        genre_counts,
        x="Genre",
        y="Movies",
        title="Most common genres in the dataset",
        color_discrete_sequence=["#d4af37"],
    )
    chart.update_layout(
        paper_bgcolor="#1e1e22",
        plot_bgcolor="#1e1e22",
        font_color="#f0ece4",
        title_font_color="#f5e6c8",
    )
    st.plotly_chart(chart, use_container_width=True)


def page_discover(engine: RecommendationEngine) -> None:
    st.header("Discover")
    st.caption("Search a movie you already know, or describe the kind of film you want.")

    st.subheader("Movie recommendation")
    search_text = st.text_input(
        "Search for a movie title",
        placeholder="Example: Inception, Finding Nemo, The Matrix",
    )
    title_choices = engine.search_titles(search_text, limit=40)
    if not title_choices:
        st.warning("No movie titles matched that search. Try a shorter name or a different spelling.")
        selected_title = st.session_state.selected_movie
    else:
        default_index = 0
        if st.session_state.selected_movie in title_choices:
            default_index = title_choices.index(st.session_state.selected_movie)
        selected_title = st.selectbox("Select a movie", title_choices, index=default_index)

    if st.button("Recommend similar movies", type="primary"):
        movie, recs = engine.recommend_similar(selected_title)
        if movie is None:
            st.warning("That movie could not be found in the dataset.")
        elif not recs:
            st.info("No similar movies were found for this title.")
        else:
            st.session_state.selected_movie = movie["title"]
            st.session_state.last_source_title = movie["title"]
            st.session_state.last_recommendations = recs

    if st.session_state.last_source_title:
        st.success(f"Selected movie: {st.session_state.last_source_title}")
        render_movie_cards(st.session_state.last_recommendations, "similar")

    st.markdown("---")
    st.subheader("What kind of movie are you looking for?")
    st.caption("NLP-based smart search. This is not ChatGPT and it does not generate new movie plots.")
    if "nlp_box" not in st.session_state:
        st.session_state.nlp_box = st.session_state.nlp_query
    nlp_query = st.text_input(
        "Describe the movie in everyday words",
        placeholder="funny adventure movie, romantic emotional movie, action science fiction...",
        key="nlp_box",
    )
    example_queries = [
        "funny adventure movie",
        "romantic emotional movie",
        "intelligent thriller",
        "action science fiction",
        "relaxing comedy",
    ]
    picked = st.radio("Quick examples", ["(type your own)"] + example_queries, horizontal=True)
    if picked != "(type your own)":
        nlp_query = picked

    if st.button("Find movies from my description"):
        if not nlp_query.strip():
            st.warning("Please enter a short description first.")
        else:
            results = engine.recommend_semantic(nlp_query)
            st.session_state.nlp_query = nlp_query
            if not results:
                st.info("No movies matched that description. Try simpler words such as comedy, action or romance.")
            else:
                st.write(f"NLP-based smart search results for: **{nlp_query}**")
                render_movie_cards(results, "nlp")


def page_mood(engine: RecommendationEngine) -> None:
    st.header("Mood-based recommendations")
    st.write("Choose how you feel. Each mood is mapped to MovieLens genres and keywords, then ranked with TF-IDF.")
    mood = st.selectbox(
        "Select a mood",
        list(MOOD_PROFILES.keys()),
        index=list(MOOD_PROFILES.keys()).index(st.session_state.get("chosen_mood", "Happy"))
        if st.session_state.get("chosen_mood", "Happy") in MOOD_PROFILES
        else 0,
    )
    profile = MOOD_PROFILES[mood]
    st.markdown(
        " ".join(f'<span class="chip">{genre}</span>' for genre in sorted(profile["genres"])),
        unsafe_allow_html=True,
    )
    if st.button("Recommend for this mood", type="primary"):
        results = engine.recommend_by_mood(mood)
        if not results:
            st.info("No mood matches were found.")
        else:
            render_movie_cards(results, f"mood-{mood}")


def page_personalized(engine: RecommendationEngine) -> None:
    st.header("Personalized recommendations")
    st.write("Pick up to 3 favourite movies. The app combines their TF-IDF content profiles.")
    titles = engine.movie_options()
    favourites = st.multiselect(
        "Your favourite movies (maximum 3)",
        options=titles,
        max_selections=3,
        placeholder="Choose 1, 2 or 3 movies",
    )
    if st.button("Create my recommendations", type="primary"):
        if not favourites:
            st.warning("Please choose at least one favourite movie.")
        else:
            selected, results = engine.recommend_personalized(favourites)
            if not selected:
                st.warning("Those titles could not be matched in the dataset.")
            elif not results:
                st.info("No personalized recommendations were found.")
            else:
                liked = ", ".join(row["title"] for row in selected)
                st.success(f"Because you liked {liked}")
                render_movie_cards(results, "personal")


def page_top_rated(engine: RecommendationEngine) -> None:
    st.header("Top rated movies")
    st.write(
        "Movies are ranked with a minimum rating-count threshold so that a film with only one or two reviews "
        "cannot appear at the top just because those few scores are high."
    )
    min_votes = st.slider("Minimum number of ratings", min_value=10, max_value=150, value=50, step=5)
    results = engine.top_rated(min_votes=min_votes, top_n=20)
    if not results:
        st.info("No movies have enough ratings for this threshold. Lower the slider and try again.")
        return
    table = pd.DataFrame(
        [
            {
                "Movie": item["title"],
                "Genre": item["genres"],
                "Average rating": item["avg_rating"],
                "Number of ratings": item["rating_count"],
            }
            for item in results
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    render_movie_cards(results, "top", show_similarity=False)


def page_genres(engine: RecommendationEngine) -> None:
    st.header("Genre explorer")
    genres = engine.all_genres()
    if not genres:
        st.warning("No genres were found in the dataset.")
        return
    genre = st.selectbox("Select a genre", genres)
    results = engine.movies_by_genre(genre, top_n=20)
    if not results:
        st.info(f"No movies were found for the {genre} genre.")
        return
    st.caption("Ordered by a rating score that prefers movies with more reviews.")
    render_movie_cards(results, f"genre-{genre}", show_similarity=False)


def page_watchlist() -> None:
    st.header("Watchlist")
    st.caption("This list exists only while the app is running. Closing the browser tab clears it.")
    watchlist = st.session_state.watchlist
    if not watchlist:
        st.info("Your watchlist is empty. Add movies from Discover, Mood, Personalized, Top Rated or Genres.")
        return

    for index, movie in enumerate(watchlist):
        left, right = st.columns([5, 1])
        with left:
            rating = rating_label(movie["avg_rating"])
            st.markdown(f"**{movie['title']}**  \n{movie['genres']}  \n{rating}")
        with right:
            if st.button("Remove", key=f"remove-{movie['movieId']}-{index}"):
                st.session_state.watchlist.pop(index)
                st.rerun()
        st.markdown("---")

    if st.button("Clear watchlist"):
        st.session_state.watchlist = []
        st.rerun()


def page_about(overview: dict) -> None:
    st.header("About this project")
    st.write(
        "Smart Movie Recommendation System is a college mini-project that demonstrates content-based filtering. "
        "It uses the **MovieLens Latest Small** dataset from GroupLens."
    )
    st.markdown(
        """
        **What it does**
        - Recommends similar movies with TF-IDF and cosine similarity
        - Understands short natural-language requests with the same TF-IDF model
        - Maps moods to genres
        - Builds a simple personal profile from up to 3 favourite movies
        - Explains each recommendation using genres and tags found in the data

        **What it does not do**
        - It is not ChatGPT or any paid AI API
        - It does not use deep learning
        - It does not use ratings inside the TF-IDF model
        """
    )
    st.info(
        f"Currently loaded: {overview['movie_count']:,} movies, "
        f"{overview['rating_count']:,} ratings and {overview['user_count']:,} users."
    )


def render_dashboard(overview: dict) -> None:
    st.sidebar.markdown("### Recommendation dashboard")
    st.sidebar.write(f"Movies: **{overview['movie_count']:,}**")
    st.sidebar.write(f"Ratings: **{overview['rating_count']:,}**")
    st.sidebar.write(f"Users: **{overview['user_count']:,}**")
    st.sidebar.write(f"Average rating: **{overview['average_rating']:.2f}**")
    selected = st.session_state.get("last_source_title") or st.session_state.get("selected_movie") or "None yet"
    st.sidebar.write(f"Selected movie: **{selected}**")
    st.sidebar.write(f"Recommendations shown: **{len(st.session_state.get('last_recommendations', []))}**")
    st.sidebar.write(f"Watchlist: **{len(st.session_state.watchlist)}**")


def main() -> None:
    st.set_page_config(
        page_title="Smart Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_state()

    st.sidebar.title("🎬 Smart Movie")
    st.sidebar.caption("Content-based discovery")
    st.sidebar.radio("Navigate", PAGES, key="nav")

    try:
        _data, engine, overview = get_system()
    except DatasetError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error("The application could not start. Please check your internet connection and try again.")
        st.caption(str(error))
        st.stop()

    render_dashboard(overview)
    page = st.session_state.nav

    if page == "🏠 Home":
        page_home(engine, overview)
    elif page == "🔍 Discover":
        page_discover(engine)
    elif page == "😊 Mood":
        page_mood(engine)
    elif page == "⭐ Personalized":
        page_personalized(engine)
    elif page == "🏆 Top Rated":
        page_top_rated(engine)
    elif page == "🎭 Genres":
        page_genres(engine)
    elif page == "🎞️ Watchlist":
        page_watchlist()
    else:
        page_about(overview)


if __name__ == "__main__":
    main()
