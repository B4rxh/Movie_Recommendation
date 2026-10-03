import pandas as pd
import pytest

from app.evaluation import evaluate, precision_at_k, recall_at_k
from app.recommender import ContentRecommender


def make_movies() -> pd.DataFrame:
    rows = [
        (1, "Space Odyssey", ["Science Fiction"], ["space"], "Astronauts explore deep space and distant planets."),
        (2, "Star Voyage", ["Science Fiction"], ["space"], "A crew travels across the galaxy to a new planet."),
        (3, "Galaxy Wars", ["Science Fiction", "Action"], ["space"], "Rebels fight an empire across the galaxy."),
        (4, "Love in Paris", ["Romance"], ["paris"], "Two strangers fall in love in Paris."),
        (5, "Paris Nights", ["Romance", "Drama"], ["paris"], "A love story unfolds in the streets of Paris."),
        (6, "Cooking Dreams", ["Comedy"], ["chef"], "A young chef opens a restaurant."),
    ]
    return pd.DataFrame(
        [
            {
                "id": movie_id,
                "title": title,
                "tagline": "",
                "overview": overview,
                "release_date": "2020-01-01",
                "genres": genres,
                "keywords": keywords,
                "cast": [],
                "director": "",
                "vote_average": 7.0,
                "vote_count": 100,
                "popularity": 10.0,
                "runtime": 100,
                "original_language": "en",
                "poster_path": None,
            }
            for movie_id, title, genres, keywords, overview in rows
        ]
    )


@pytest.fixture
def model() -> ContentRecommender:
    return ContentRecommender().fit(make_movies())


def test_similar_movies_rank_first(model):
    result = model.recommend(1, top_n=2, quality_weight=0.0)
    assert set(result["id"]) == {2, 3}


def test_query_movie_is_excluded(model):
    assert 1 not in model.recommend(1, top_n=5)["id"].tolist()


def test_unknown_movie_gets_recommendations(model):
    new_movie = {
        "id": 999,
        "title": "Future Space Saga",
        "overview": "Astronauts cross the galaxy.",
        "genres": ["Science Fiction"],
        "keywords": ["space"],
        "cast": [],
        "director": "",
        "tagline": "",
    }
    result = model.recommend_for_movie(new_movie, top_n=3, quality_weight=0.0)
    assert result.iloc[0]["id"] in {1, 2, 3}


def test_personalised_recommendations_exclude_liked(model):
    liked = [model.movies.iloc[3].to_dict()]
    result = model.recommend_for_movies(liked, top_n=3, quality_weight=0.0)
    assert 4 not in result["id"].tolist()
    assert result.iloc[0]["id"] == 5


def test_unknown_id_raises(model):
    with pytest.raises(KeyError):
        model.recommend(12345)


def test_save_and_load_round_trip(model, tmp_path):
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = ContentRecommender.load(path)
    assert loaded.recommend(1, 2, 0.0)["id"].tolist() == model.recommend(1, 2, 0.0)["id"].tolist()


def test_metric_functions():
    assert precision_at_k([1, 2, 3, 4], {1, 3}, 4) == 0.5
    assert recall_at_k([1, 2, 3, 4], {1, 3, 9, 10}, 4) == 0.5


def test_evaluate_returns_expected_keys(model):
    metrics = evaluate(model, k=2, sample_size=5)
    assert {"precision_at_k", "recall_at_k", "diversity", "coverage"} <= set(metrics)
    assert metrics["precision_at_k"] > 0
