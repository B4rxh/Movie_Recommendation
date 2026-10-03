import random
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.recommender import ContentRecommender
from app.utils import as_list


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def genre_jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def build_relevance_sets(recommender: ContentRecommender, min_jaccard: float = 0.5) -> dict[int, set[int]]:
    ids = recommender.movies["id"].astype(int).tolist()
    genres = [set(as_list(value)) for value in recommender.movies["genres"]]
    relevance: dict[int, set[int]] = {}
    for i, movie_id in enumerate(ids):
        relevance[movie_id] = {
            ids[j] for j in range(len(ids)) if j != i and genre_jaccard(genres[i], genres[j]) >= min_jaccard
        }
    return relevance


def diversity_of(recommender: ContentRecommender, movie_ids: Sequence[int]) -> float:
    if len(movie_ids) < 2:
        return 0.0
    positions = [recommender.index_of(movie_id) for movie_id in movie_ids]
    similarity = cosine_similarity(recommender.matrix[positions])
    count = len(positions)
    mean_pairwise = (similarity.sum() - np.trace(similarity)) / (count * (count - 1))
    return float(1.0 - mean_pairwise)


def evaluate(
    recommender: ContentRecommender,
    k: int = 10,
    sample_size: int = 100,
    min_jaccard: float = 0.5,
    quality_weight: float = 0.0,
    seed: int = 42,
) -> dict[str, float]:
    relevance = build_relevance_sets(recommender, min_jaccard)
    candidates = [movie_id for movie_id, relevant in relevance.items() if relevant]
    random.Random(seed).shuffle(candidates)
    sample = candidates[:sample_size]

    precisions, recalls, diversities = [], [], []
    recommended_everywhere: set[int] = set()

    for movie_id in sample:
        result = recommender.recommend(movie_id, top_n=k, quality_weight=quality_weight)
        recommended = result["id"].astype(int).tolist()
        recommended_everywhere.update(recommended)
        precisions.append(precision_at_k(recommended, relevance[movie_id], k))
        recalls.append(recall_at_k(recommended, relevance[movie_id], k))
        diversities.append(diversity_of(recommender, recommended))

    if not sample:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "diversity": 0.0, "coverage": 0.0, "evaluated": 0}

    return {
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "diversity": round(float(np.mean(diversities)), 4),
        "coverage": round(len(recommended_everywhere) / len(recommender), 4),
        "evaluated": len(sample),
    }
