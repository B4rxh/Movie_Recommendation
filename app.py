from typing import Any

import pandas as pd
import streamlit as st

from app import config
from app.api import TMDBClient, TMDBError
from app.data_loader import flatten_movie
from app.recommender import ContentRecommender
from app.utils import as_list, poster_url, year_from_date

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_recommender() -> ContentRecommender | None:
    if not config.MODEL_PATH.exists():
        return None
    return ContentRecommender.load(config.MODEL_PATH)


@st.cache_resource
def get_client() -> TMDBClient:
    return TMDBClient()


def show_poster(poster_path: str | None) -> None:
    url = poster_url(poster_path)
    if url:
        st.image(url, use_container_width=True)
    else:
        st.caption("No poster available")


def render_details(movie: dict[str, Any]) -> None:
    poster_col, info_col = st.columns([1, 3])
    with poster_col:
        show_poster(movie.get("poster_path"))
    with info_col:
        st.subheader(f"{movie['title']} ({year_from_date(movie.get('release_date'))})")
        if movie.get("tagline"):
            st.caption(movie["tagline"])
        st.write(" · ".join(as_list(movie.get("genres"))) or "No genres listed")
        rating_col, votes_col, runtime_col, date_col = st.columns(4)
        rating_col.metric("Rating", f"{movie.get('vote_average', 0):.1f}/10")
        votes_col.metric("Votes", f"{movie.get('vote_count', 0):,}")
        runtime_col.metric("Runtime", f"{movie.get('runtime', 0)} min")
        date_col.metric("Released", movie.get("release_date") or "N/A")
        st.write(movie.get("overview") or "No overview available.")
        if movie.get("director"):
            st.caption(f"Director: {movie['director']}")
        if movie.get("cast"):
            st.caption("Cast: " + ", ".join(movie["cast"]))


def render_recommendations(recommendations: pd.DataFrame, columns_per_row: int = 5) -> None:
    if recommendations.empty:
        st.info("No similar movies found. Try rebuilding the dataset with more pages.")
        return
    rows = [recommendations.iloc[i : i + columns_per_row] for i in range(0, len(recommendations), columns_per_row)]
    for row in rows:
        columns = st.columns(columns_per_row)
        for column, (_, movie) in zip(columns, row.iterrows()):
            with column:
                show_poster(movie["poster_path"])
                st.markdown(f"**{movie['title']}**")
                st.caption(f"{year_from_date(movie['release_date'])} · ⭐ {movie['vote_average']:.1f}")
                st.caption(" · ".join(as_list(movie["genres"])))
                st.caption(f"Match: {movie['similarity']:.0%}")
                with st.expander("Overview"):
                    st.write(movie["overview"] or "No overview available.")


def search_section(client: TMDBClient) -> dict[str, Any] | None:
    query = st.text_input("Search for a movie", placeholder="e.g. Interstellar").strip()
    if not query:
        return None
    results = client.search_movies(query)
    if not results:
        st.warning("No movies found for that search.")
        return None
    labels = {movie["id"]: f"{movie['title']} ({year_from_date(movie.get('release_date'))})" for movie in results[:10]}
    chosen_id = st.selectbox("Select a movie", options=list(labels), format_func=labels.get)
    return flatten_movie(client.get_movie_details(chosen_id))


def main() -> None:
    st.title("🎬 Movie Recommender")

    recommender = load_recommender()
    if recommender is None:
        st.error("No trained model found. Run `python -m scripts.build_dataset` first.")
        st.stop()

    try:
        client = get_client()
    except TMDBError as exc:
        st.error(str(exc))
        st.stop()

    top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
    quality_weight = st.sidebar.slider("Popularity & rating influence", 0.0, 0.5, 0.15, 0.05)
    st.sidebar.caption(f"{len(recommender)} movies indexed")

    liked: dict[int, dict[str, Any]] = st.session_state.setdefault("liked", {})

    try:
        selected = search_section(client)
    except TMDBError as exc:
        st.error(f"TMDB problem: {exc}")
        selected = None

    similar_tab, taste_tab = st.tabs(["Similar to this movie", "Based on my list"])

    with similar_tab:
        if selected is None:
            st.info("Search for a movie above to get recommendations.")
        else:
            render_details(selected)
            if st.button("Add to my list"):
                liked[selected["id"]] = selected
            if selected["id"] not in recommender:
                st.caption("This movie is newer than our dataset, so its profile was built on the fly.")
            st.divider()
            st.subheader("You might also like")
            render_recommendations(recommender.recommend_for_movie(selected, top_n, quality_weight))

    with taste_tab:
        if not liked:
            st.info("Add movies to your list from the first tab to get personalised picks.")
        else:
            st.write("Your list: " + ", ".join(movie["title"] for movie in liked.values()))
            if st.button("Clear my list"):
                liked.clear()
                st.rerun()
            st.subheader("Picked for your taste")
            render_recommendations(recommender.recommend_for_movies(list(liked.values()), top_n, quality_weight))


main()
