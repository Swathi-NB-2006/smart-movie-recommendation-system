"""
Load and prepare the MovieLens small dataset.

If the CSV files are missing, this module downloads the official
MovieLens Latest Small zip file and extracts it into the data/ folder.
"""

from __future__ import annotations

import io
import re
import ssl
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "ml-latest-small"

MOVIELENS_URLS = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
)

REQUIRED_FILES = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")


class DatasetError(Exception):
    """Raised when the MovieLens dataset cannot be downloaded or loaded."""


def _dataset_ready() -> bool:
    return all((DATASET_DIR / name).exists() for name in REQUIRED_FILES)


def _ssl_contexts() -> list[ssl.SSLContext | None]:
    contexts: list[ssl.SSLContext | None] = []
    try:
        import certifi

        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    contexts.append(ssl.create_default_context())
    contexts.append(ssl._create_unverified_context())
    contexts.append(None)
    return contexts


def _download_zip(url: str, timeout: int = 90) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "SmartMovieRecommendation/1.0 (educational project)"},
    )
    last_error: Exception | None = None
    for context in _ssl_contexts():
        try:
            with urlopen(request, timeout=timeout, context=context) as response:
                return response.read()
        except (URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
    raise last_error or DatasetError("Download failed for an unknown reason.")


def download_movielens() -> None:
    """Download and extract MovieLens small if it is not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _dataset_ready():
        return

    last_error: Exception | None = None
    zip_bytes = None
    for url in MOVIELENS_URLS:
        try:
            zip_bytes = _download_zip(url)
            break
        except (URLError, TimeoutError, OSError, DatasetError) as error:
            last_error = error

    if zip_bytes is None:
        raise DatasetError(
            "Could not download the MovieLens dataset. "
            "Please check your internet connection and try again. "
            f"Details: {last_error}"
        )

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            archive.extractall(DATA_DIR)
    except zipfile.BadZipFile as error:
        raise DatasetError(
            "The downloaded MovieLens file was not a valid zip archive."
        ) from error

    if not _dataset_ready():
        raise DatasetError(
            "The MovieLens zip extracted, but expected CSV files were not found."
        )


def _parse_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    if not match:
        return None
    return int(match.group(1))


def _clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def _prepare_movies(movies: pd.DataFrame, tags: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies["movieId"] = movies["movieId"].astype(int)
    movies["title"] = movies["title"].fillna("Unknown Title")
    movies["genres"] = movies["genres"].fillna("(no genres listed)")

    tag_text = (
        tags.dropna(subset=["tag"])
        .assign(tag=lambda frame: frame["tag"].astype(str).str.lower().str.strip())
        .groupby("movieId")["tag"]
        .apply(lambda values: " ".join(sorted(set(values))))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )

    rating_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )

    prepared = movies.merge(tag_text, on="movieId", how="left")
    prepared = prepared.merge(rating_stats, on="movieId", how="left")
    prepared["tags_text"] = prepared["tags_text"].fillna("")
    prepared["avg_rating"] = prepared["avg_rating"].astype(float)
    prepared["rating_count"] = prepared["rating_count"].fillna(0).astype(int)

    prepared["year"] = prepared["title"].map(_parse_year)
    prepared["clean_title"] = prepared["title"].map(_clean_title)
    prepared["genre_list"] = prepared["genres"].apply(
        lambda value: [item for item in str(value).split("|") if item and item != "(no genres listed)"]
    )
    prepared["genres_text"] = prepared["genre_list"].apply(lambda items: " ".join(items))

    # Title + genres + tags form the content used by TF-IDF.
    # Ratings are intentionally excluded from this text.
    prepared["content"] = (
        prepared["clean_title"].fillna("")
        + " "
        + prepared["genres_text"]
        + " "
        + prepared["genres_text"]
        + " "
        + prepared["tags_text"]
    ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    prepared["search_blob"] = (
        prepared["title"].fillna("").str.lower()
        + " "
        + prepared["clean_title"].fillna("").str.lower()
    )
    return prepared.sort_values("title").reset_index(drop=True)


def load_dataset() -> dict[str, pd.DataFrame]:
    """
    Ensure MovieLens files exist, then return prepared data frames.

    Returns a dictionary with:
    - movies: enriched movie table used by the recommender
    - ratings: raw ratings
    - tags: raw tags
    - links: IMDb / TMDb identifiers
    """
    download_movielens()

    movies_raw = pd.read_csv(DATASET_DIR / "movies.csv")
    ratings = pd.read_csv(DATASET_DIR / "ratings.csv")
    tags = pd.read_csv(DATASET_DIR / "tags.csv")
    links = pd.read_csv(DATASET_DIR / "links.csv")

    movies = _prepare_movies(movies_raw, tags, ratings)
    movies = movies.merge(links, on="movieId", how="left")
    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
    }


def dataset_overview(movies: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, float | int]:
    """Summary numbers for the dashboard."""
    return {
        "movie_count": int(len(movies)),
        "rating_count": int(len(ratings)),
        "user_count": int(ratings["userId"].nunique()) if not ratings.empty else 0,
        "average_rating": float(ratings["rating"].mean()) if not ratings.empty else 0.0,
        "genre_count": int(movies["genre_list"].explode().dropna().nunique()),
    }
