import re
from typing import Any, Iterable

from app.config import IMAGE_BASE_URL


def clean_text(text: str | None) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def squash(names: Iterable[str]) -> list[str]:
    squashed = (re.sub(r"[^a-z0-9]", "", str(name).lower()) for name in names)
    return [name for name in squashed if name]


def as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def poster_url(poster_path: str | None) -> str | None:
    if not isinstance(poster_path, str) or not poster_path:
        return None
    return f"{IMAGE_BASE_URL}{poster_path}"


def year_from_date(release_date: str | None) -> str:
    if isinstance(release_date, str) and len(release_date) >= 4:
        return release_date[:4]
    return "N/A"
