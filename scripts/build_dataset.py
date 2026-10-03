import argparse

from app import config
from app.api import TMDBClient, TMDBError
from app.data_loader import collect_movies, load_dataset, merge_movies, save_dataset
from app.evaluation import evaluate
from app.recommender import ContentRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch movies from TMDB and build the recommender.")
    parser.add_argument("--pages", type=int, default=5, help="pages per TMDB category (20 movies per page)")
    parser.add_argument("--categories", nargs="+", default=list(TMDBClient.CATEGORIES))
    parser.add_argument("--rebuild", action="store_true", help="ignore the existing dataset and start fresh")
    parser.add_argument("--evaluate", action="store_true", help="print evaluation metrics after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    try:
        client = TMDBClient()
    except TMDBError as exc:
        raise SystemExit(str(exc))

    existing = None
    if config.DATASET_PATH.exists() and not args.rebuild:
        existing = load_dataset()
        print(f"Existing dataset: {len(existing)} movies")

    skip_ids = existing["id"].tolist() if existing is not None else []

    print("Fetching movies from TMDB...")
    try:
        new_movies = collect_movies(client, args.categories, args.pages, skip_ids)
    except TMDBError as exc:
        raise SystemExit(f"Could not fetch movies: {exc}")
    print(f"Added {len(new_movies)} new movies")

    movies = merge_movies(existing, new_movies)
    if movies.empty:
        raise SystemExit("No movies available to train on.")

    save_dataset(movies)
    model = ContentRecommender().fit(movies)
    model.save(config.MODEL_PATH)
    print(f"Saved dataset and model: {len(model)} movies, {model.matrix.shape[1]} TF-IDF features")

    if args.evaluate:
        print("Evaluation:", evaluate(model))


if __name__ == "__main__":
    main()
