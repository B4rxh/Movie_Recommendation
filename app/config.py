import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

RAW_DIR = BASE_DIR / "data" / "raw"
CACHE_DIR = RAW_DIR / "cache"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = PROCESSED_DIR / "movies.json"
MODEL_PATH = MODELS_DIR / "recommender.joblib"

REQUEST_TIMEOUT = 10
MAX_RETRIES = 4

SEARCH_TTL = 60 * 60
LIST_TTL = 6 * 60 * 60
DETAILS_TTL = 7 * 24 * 60 * 60


def ensure_dirs() -> None:
    for folder in (CACHE_DIR, PROCESSED_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
