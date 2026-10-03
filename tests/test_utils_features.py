from app.features import build_profile
from app.utils import clean_text, poster_url, squash, year_from_date


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, World!!  It's 2024.") == "hello world it s 2024"


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_squash_joins_multiword_names():
    assert squash(["Science Fiction", "Christopher Nolan"]) == ["sciencefiction", "christophernolan"]


def test_poster_url_and_year():
    assert poster_url("/abc.jpg").endswith("/abc.jpg")
    assert poster_url(None) is None
    assert year_from_date("2014-11-05") == "2014"
    assert year_from_date("") == "N/A"


def test_build_profile_contains_weighted_tokens():
    movie = {
        "genres": ["Science Fiction"],
        "keywords": ["space travel"],
        "cast": ["Matthew McConaughey"],
        "director": "Christopher Nolan",
        "overview": "A team travels through a wormhole.",
        "tagline": "",
    }
    profile = build_profile(movie)
    assert profile.count("sciencefiction") == 3
    assert profile.count("christophernolan") == 2
    assert "wormhole" in profile
