import pytest

from app.api import TMDBClient, TMDBError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.api.time.sleep", lambda seconds: None)


def make_client(tmp_path, responses):
    client = TMDBClient(api_key="test-key", cache_dir=tmp_path)
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return responses.pop(0)

    client.session.get = fake_get
    return client, calls


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.config.TMDB_API_KEY", "")
    with pytest.raises(TMDBError):
        TMDBClient(api_key="", cache_dir=tmp_path)


def test_search_is_cached(tmp_path):
    client, calls = make_client(tmp_path, [FakeResponse(200, {"results": [{"id": 1, "title": "A"}]})])
    assert client.search_movies("a")[0]["id"] == 1
    assert client.search_movies("a")[0]["id"] == 1
    assert len(calls) == 1


def test_not_found_raises(tmp_path):
    client, _ = make_client(tmp_path, [FakeResponse(404)])
    with pytest.raises(TMDBError):
        client.get_movie_details(1)


def test_rate_limit_is_retried(tmp_path):
    responses = [FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(200, {"results": []})]
    client, calls = make_client(tmp_path, responses)
    assert client.search_movies("x") == []
    assert len(calls) == 2


def test_gives_up_after_max_retries(tmp_path):
    client, _ = make_client(tmp_path, [FakeResponse(500) for _ in range(10)])
    with pytest.raises(TMDBError):
        client.search_movies("x")
