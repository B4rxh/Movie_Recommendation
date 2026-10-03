import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app import config
from app.api import TMDBClient, TMDBError


def flatten_movie(details: dict[str, Any]) -> dict[str, Any]:
    credits = details.get("credits") or {}
    cast = [person["name"] for person in credits.get("cast", [])[:5]]
    directors = [person["name"] for person in credits.get("crew", []) if person.get("job") == "Director"]
    keywords = [item["name"] for item in (details.get("keywords") or {}).get("keywords", [])]

    return {
        "id": int(details["id"]),
        "title": details.get("title") or "",
        "tagline": details.get("tagline") or "",
        "overview": details.get("overview") or "",
        "release_date": details.get("release_date") or "",
        "genres": [genre["name"] for genre in details.get("genres", [])],
        "keywords": keywords,
        "cast": cast,
        "director": directors[0] if directors else "",
        "vote_average": float(details.get("vote_average") or 0),
        "vote_count": int(details.get("vote_count") or 0),
        "popularity": float(details.get("popularity") or 0),
        "runtime": int(details.get("runtime") or 0),
        "original_language": details.get("original_language") or "",
        "poster_path": details.get("poster_path"),
    }


def is_usable(movie: dict[str, Any]) -> bool:
    has_text = bool(movie["overview"].strip()) or bool(movie["keywords"])
    return bool(movie["title"]) and has_text


def collect_movies(
    client: TMDBClient,
    categories: Iterable[str],
    pages: int,
    skip_ids: Iterable[int] = (),
    workers: int = 8,
) -> list[dict[str, Any]]:
    found_ids: set[int] = set()
    for category in categories:
        found_ids.update(client.list_movie_ids(category, pages))

    new_ids = sorted(found_ids - set(skip_ids))

    def fetch(movie_id: int) -> dict[str, Any] | None:
        try:
            return flatten_movie(client.get_movie_details(movie_id))
        except TMDBError:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(fetch, new_ids))

    return [movie for movie in fetched if movie and is_usable(movie)]


def merge_movies(existing: pd.DataFrame | None, new_movies: list[dict[str, Any]]) -> pd.DataFrame:
    new_frame = pd.DataFrame(new_movies)
    if existing is None or existing.empty:
        return new_frame.reset_index(drop=True)
    if new_frame.empty:
        return existing.reset_index(drop=True)
    combined = pd.concat([existing, new_frame], ignore_index=True)
    return combined.drop_duplicates(subset="id", keep="last").reset_index(drop=True)


def save_dataset(movies: pd.DataFrame, path: Path = config.DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(movies.to_dict("records")), encoding="utf-8")


def load_dataset(path: Path = config.DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}. Run: python -m scripts.build_dataset")
    records = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(records)
