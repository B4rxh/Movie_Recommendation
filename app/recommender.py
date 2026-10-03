from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.features import add_profiles, build_profile


def compute_quality(movies: pd.DataFrame) -> np.ndarray:
    rated = movies[movies["vote_count"] > 0]
    mean_rating = rated["vote_average"].mean() if not rated.empty else 0.0
    min_votes = movies["vote_count"].quantile(0.6)

    votes = movies["vote_count"].to_numpy(dtype=float)
    ratings = movies["vote_average"].to_numpy(dtype=float)
    weighted = (votes / (votes + min_votes + 1e-9)) * ratings + (min_votes / (votes + min_votes + 1e-9)) * mean_rating
    weighted_norm = weighted / 10.0

    log_popularity = np.log1p(movies["popularity"].to_numpy(dtype=float))
    popularity_norm = log_popularity / log_popularity.max() if log_popularity.max() > 0 else log_popularity

    return 0.7 * weighted_norm + 0.3 * popularity_norm


class ContentRecommender:
    def __init__(self, max_features: int = 30000) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, sublinear_tf=True)
        self.movies: pd.DataFrame = pd.DataFrame()
        self.matrix: csr_matrix | None = None
        self._index: dict[int, int] = {}

    def fit(self, movies: pd.DataFrame) -> "ContentRecommender":
        frame = add_profiles(movies.reset_index(drop=True))
        self.matrix = self.vectorizer.fit_transform(frame["profile"])
        frame["quality"] = compute_quality(frame)
        self.movies = frame
        self._index = {int(movie_id): position for position, movie_id in enumerate(frame["id"])}
        return self

    def __contains__(self, movie_id: int) -> bool:
        return int(movie_id) in self._index

    def __len__(self) -> int:
        return len(self.movies)

    def index_of(self, movie_id: int) -> int:
        try:
            return self._index[int(movie_id)]
        except KeyError:
            raise KeyError(f"Movie id {movie_id} is not in the model") from None

    def _vector_for(self, movie: Mapping[str, Any]) -> csr_matrix:
        movie_id = movie.get("id")
        if movie_id is not None and movie_id in self:
            return self.matrix[self.index_of(movie_id)]
        return self.vectorizer.transform([build_profile(movie)])

    def _rank(self, query_vector: Any, exclude_ids: set[int], top_n: int, quality_weight: float) -> pd.DataFrame:
        similarity = cosine_similarity(query_vector, self.matrix).ravel()
        score = (1.0 - quality_weight) * similarity + quality_weight * self.movies["quality"].to_numpy()
        score[similarity <= 0] = -np.inf
        for movie_id in exclude_ids:
            if movie_id in self:
                score[self.index_of(movie_id)] = -np.inf

        top = np.argsort(-score)[:top_n]
        top = top[np.isfinite(score[top])]

        result = self.movies.iloc[top].copy()
        result["similarity"] = similarity[top]
        result["score"] = score[top]
        return result.reset_index(drop=True)

    def recommend(self, movie_id: int, top_n: int = 10, quality_weight: float = 0.15) -> pd.DataFrame:
        vector = self.matrix[self.index_of(movie_id)]
        return self._rank(vector, {int(movie_id)}, top_n, quality_weight)

    def recommend_for_movie(self, movie: Mapping[str, Any], top_n: int = 10, quality_weight: float = 0.15) -> pd.DataFrame:
        exclude = {int(movie["id"])} if movie.get("id") is not None else set()
        return self._rank(self._vector_for(movie), exclude, top_n, quality_weight)

    def recommend_for_movies(
        self, movies: Sequence[Mapping[str, Any]], top_n: int = 10, quality_weight: float = 0.15
    ) -> pd.DataFrame:
        if not movies:
            return self.movies.iloc[0:0].copy()
        stacked = vstack([self._vector_for(movie) for movie in movies])
        taste_vector = np.asarray(stacked.mean(axis=0))
        exclude = {int(movie["id"]) for movie in movies if movie.get("id") is not None}
        return self._rank(taste_vector, exclude, top_n, quality_weight)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "movies": self.movies}, path)

    @classmethod
    def load(cls, path: Path) -> "ContentRecommender":
        payload = joblib.load(path)
        model = cls()
        model.vectorizer = payload["vectorizer"]
        model.matrix = payload["matrix"]
        model.movies = payload["movies"]
        model._index = {int(movie_id): position for position, movie_id in enumerate(model.movies["id"])}
        return model
