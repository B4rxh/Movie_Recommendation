import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from app import config


class TMDBError(Exception):
    pass


class TMDBClient:
    CATEGORIES = ("popular", "top_rated", "now_playing", "upcoming")

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None) -> None:
        self.api_key = (api_key or config.TMDB_API_KEY).strip()
        if not self.api_key:
            raise TMDBError("TMDB_API_KEY is missing. Add it to your .env file.")

        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"accept": "application/json"})
        self._auth_params: dict[str, str] = {}
        if self.api_key.startswith("eyJ"):
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            self._auth_params["api_key"] = self.api_key

    def _cache_file(self, path: str, params: dict[str, Any]) -> Path:
        raw_key = json.dumps([path, params], sort_keys=True)
        digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _read_cache(cache_file: Path, ttl: float | None) -> Any | None:
        if not cache_file.exists():
            return None
        if ttl is not None and time.time() - cache_file.stat().st_mtime > ttl:
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _retry_delay(response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            if str(retry_after).isdigit():
                return float(retry_after)
        return 0.5 * (2 ** attempt)

    def _get(self, path: str, params: dict[str, Any] | None = None, ttl: float = config.DETAILS_TTL) -> Any:
        params = params or {}
        cache_file = self._cache_file(path, params)

        cached = self._read_cache(cache_file, ttl)
        if cached is not None:
            return cached

        url = f"{config.TMDB_BASE_URL}{path}"
        last_problem = "unknown error"

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    url,
                    params={**params, **self._auth_params},
                    timeout=config.REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_problem = str(exc)
                time.sleep(self._retry_delay(None, attempt))
                continue

            if response.status_code == 200:
                data = response.json()
                cache_file.write_text(json.dumps(data), encoding="utf-8")
                return data
            if response.status_code == 401:
                raise TMDBError("TMDB rejected the API key (401). Check TMDB_API_KEY in .env.")
            if response.status_code == 404:
                raise TMDBError(f"TMDB could not find {path} (404).")
            if response.status_code == 429 or response.status_code >= 500:
                last_problem = f"HTTP {response.status_code}"
                time.sleep(self._retry_delay(response, attempt))
                continue
            raise TMDBError(f"TMDB returned HTTP {response.status_code} for {path}.")

        stale = self._read_cache(cache_file, None)
        if stale is not None:
            return stale
        raise TMDBError(f"TMDB request failed after {config.MAX_RETRIES} attempts: {last_problem}")

    def search_movies(self, query: str, page: int = 1) -> list[dict[str, Any]]:
        params = {"query": query, "include_adult": "false", "language": "en-US", "page": page}
        data = self._get("/search/movie", params, ttl=config.SEARCH_TTL)
        return data.get("results", [])

    def get_movie_details(self, movie_id: int) -> dict[str, Any]:
        params = {"append_to_response": "credits,keywords", "language": "en-US"}
        return self._get(f"/movie/{int(movie_id)}", params, ttl=config.DETAILS_TTL)

    def list_movie_ids(self, category: str, pages: int = 1) -> list[int]:
        if category not in self.CATEGORIES:
            raise TMDBError(f"Unknown category '{category}'. Choose from {self.CATEGORIES}.")
        movie_ids: list[int] = []
        for page in range(1, pages + 1):
            data = self._get(f"/movie/{category}", {"page": page, "language": "en-US"}, ttl=config.LIST_TTL)
            movie_ids.extend(movie["id"] for movie in data.get("results", []))
            if page >= data.get("total_pages", page):
                break
        return movie_ids
