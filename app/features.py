from typing import Any, Mapping

import pandas as pd

from app.utils import as_list, clean_text, squash

FEATURE_REPEATS = {"genres": 3, "keywords": 2, "director": 2, "cast": 1}


def build_profile(movie: Mapping[str, Any]) -> str:
    director = movie.get("director")
    director_tokens = squash([director]) if isinstance(director, str) else []

    tokens: list[str] = []
    tokens += squash(as_list(movie.get("genres"))) * FEATURE_REPEATS["genres"]
    tokens += squash(as_list(movie.get("keywords"))) * FEATURE_REPEATS["keywords"]
    tokens += director_tokens * FEATURE_REPEATS["director"]
    tokens += squash(as_list(movie.get("cast"))) * FEATURE_REPEATS["cast"]
    tokens.append(clean_text(movie.get("overview")))
    tokens.append(clean_text(movie.get("tagline")))
    return " ".join(token for token in tokens if token)


def add_profiles(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies["profile"] = movies.apply(build_profile, axis=1)
    return movies
