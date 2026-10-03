# Movie Recommendation System

A content-based movie recommender built on live data from the TMDB API, with a Streamlit interface.

## Features

- Live movie search through TMDB
- Posters, genres, ratings, release dates and overviews
- TF-IDF and cosine similarity over genres, keywords, cast, director and overview
- Ranking that blends similarity with a weighted rating and popularity score
- Works for movies that are not in the local dataset
- Personalised picks from a list of movies you choose
- Cached API calls with retries and rate-limit handling
- Evaluation with Precision@K, Recall@K, diversity and coverage

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Put your TMDB key in `.env`, then build the dataset and start the app:

```powershell
python -m scripts.build_dataset --pages 5 --evaluate
streamlit run app.py
```

Run the build script again whenever you want newly released movies added.

## Tests

```powershell
pytest
```

## Structure

- `app/api.py` TMDB client with caching, retries and error handling
- `app/data_loader.py` turns API JSON into a clean table
- `app/features.py` builds the text profile for each movie
- `app/recommender.py` TF-IDF, similarity and ranking
- `app/evaluation.py` offline metrics
- `scripts/build_dataset.py` fetches data and trains the model
- `app.py` Streamlit interface

## Ideas for later

Sentence-transformer embeddings, a hybrid model with collaborative filtering, and evaluation with real user ratings.
