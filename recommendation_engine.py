"""
Content-based movie recommendation engine.

Pipeline:
    movie data → content text → TF-IDF vectors → cosine similarity → ranked results

Ratings are used only for statistics and top-rated lists, never inside TF-IDF.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_engine import SemanticMovieEngine

MOOD_PROFILES = {
    "Happy": {
        "query": "comedy animation family musical feel good fun uplifting happy",
        "genres": {"Comedy", "Animation", "Family", "Musical"},
    },
    "Romantic": {
        "query": "romance romantic love drama relationship emotional",
        "genres": {"Romance", "Drama"},
    },
    "Thriller": {
        "query": "thriller mystery crime suspense horror tense",
        "genres": {"Thriller", "Mystery", "Horror", "Crime"},
    },
    "Action": {
        "query": "action adventure war western fight explosion",
        "genres": {"Action", "Adventure", "War", "Western"},
    },
    "Relaxing": {
        "query": "documentary drama musical children calm gentle relaxing",
        "genres": {"Documentary", "Drama", "Musical", "Children"},
    },
    "Funny": {
        "query": "comedy funny humorous laugh parody",
        "genres": {"Comedy"},
    },
    "Intelligent": {
        "query": "mystery documentary drama sci-fi film-noir intelligent thoughtful",
        "genres": {"Mystery", "Documentary", "Drama", "Sci-Fi", "Film-Noir"},
    },
}


WORD_TO_GENRES = {
    "funny": ["Comedy"],
    "comedy": ["Comedy"],
    "humorous": ["Comedy"],
    "romantic": ["Romance"],
    "romance": ["Romance"],
    "love": ["Romance"],
    "emotional": ["Drama", "Romance"],
    "action": ["Action"],
    "adventure": ["Adventure"],
    "thriller": ["Thriller"],
    "intelligent": ["Mystery", "Drama", "Documentary", "Sci-Fi"],
    "science": ["Sci-Fi"],
    "fiction": ["Sci-Fi"],
    "scifi": ["Sci-Fi"],
    "relaxing": ["Drama", "Documentary", "Children", "Musical"],
    "family": ["Family", "Children", "Animation"],
    "scary": ["Horror"],
    "horror": ["Horror"],
    "animation": ["Animation"],
}

QUERY_SYNONYMS = {
    "funny": "comedy humorous laugh parody",
    "comedy": "comedy funny humorous",
    "romantic": "romance love relationship drama",
    "emotional": "drama emotional romance",
    "intelligent": "mystery documentary thoughtful sci-fi",
    "thriller": "thriller mystery suspense crime",
    "action": "action adventure",
    "science": "sci-fi science fiction",
    "fiction": "sci-fi science fiction",
    "relaxing": "calm gentle drama documentary musical",
    "adventure": "adventure action children animation",
}


class RecommendationEngine:
    """TF-IDF + cosine similarity recommender for MovieLens movies."""

    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        self.id_to_index = {int(movie_id): idx for idx, movie_id in enumerate(self.movies["movieId"])}

        corpus = self.movies["content"].fillna("").tolist()
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=8000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.global_mean_rating = float(self.movies["avg_rating"].mean(skipna=True) or 0.0)
        self.semantic_engine = SemanticMovieEngine(self.movies)

    def movie_options(self) -> list[str]:
        return self.movies["title"].tolist()

    def all_genres(self) -> list[str]:
        genres = sorted({genre for row in self.movies["genre_list"] for genre in row})
        return genres

    def find_movie(self, title: str) -> pd.Series | None:
        if not title:
            return None
        exact = self.movies[self.movies["title"] == title]
        if not exact.empty:
            return exact.iloc[0]

        lowered = title.strip().lower()
        contains = self.movies[self.movies["search_blob"].str.contains(re.escape(lowered), na=False)]
        if contains.empty:
            return None
        return contains.iloc[0]

    def search_titles(self, query: str, limit: int = 25) -> list[str]:
        if not query or not query.strip():
            return self.movies["title"].head(limit).tolist()
        lowered = query.strip().lower()
        matches = self.movies[self.movies["search_blob"].str.contains(re.escape(lowered), na=False)]
        return matches["title"].head(limit).tolist()

    def _row_to_result(self, row: pd.Series, similarity: float, reason: str) -> dict:
        avg_rating = row["avg_rating"]
        rating_display = None if pd.isna(avg_rating) or row["rating_count"] == 0 else round(float(avg_rating), 2)
        return {
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": ", ".join(row["genre_list"]) if row["genre_list"] else "Not listed",
            "genre_list": list(row["genre_list"]),
            "avg_rating": rating_display,
            "rating_count": int(row["rating_count"]),
            "similarity": round(max(0.0, float(similarity)) * 100, 1),
            "reason": reason,
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "tags_text": row["tags_text"],
        }

    def _explain_overlap(self, source_genres: set[str], source_tags: set[str], candidate: pd.Series) -> str:
        candidate_genres = set(candidate["genre_list"])
        shared_genres = sorted(source_genres & candidate_genres)

        candidate_tags = set(str(candidate["tags_text"]).split()) if candidate["tags_text"] else set()
        shared_tags = sorted(source_tags & candidate_tags)[:4]

        parts = []
        if shared_genres:
            if len(shared_genres) == 1:
                parts.append(f"it shares {shared_genres[0]} characteristics")
            else:
                parts.append("it shares " + " and ".join([", ".join(shared_genres[:-1]), shared_genres[-1]]) + " characteristics")
        if shared_tags:
            parts.append("similar tags such as " + ", ".join(shared_tags))

        if not parts:
            return "Recommended because its overall title, genre and tag content is similar."
        return "Recommended because " + " and ".join(parts) + "."

    def _explain_query(self, query: str, candidate: pd.Series) -> str:
        tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2}
        genre_tokens = {genre.lower() for genre in candidate["genre_list"]}
        tag_tokens = set(str(candidate["tags_text"]).lower().split()) if candidate["tags_text"] else set()
        matched_genres = sorted(token.title() for token in tokens if token in genre_tokens or token + "s" in genre_tokens)
        # Map common words to genres
        mapped = []
        for token in tokens:
            for genre in WORD_TO_GENRES.get(token, []):
                if genre in set(candidate["genre_list"]) and genre not in mapped:
                    mapped.append(genre)
        labels = []
        for item in mapped + matched_genres:
            if item not in labels:
                labels.append(item)

        matched_tags = sorted(token for token in tokens if token in tag_tokens)[:4]
        parts = []
        if labels:
            parts.append("it matches " + " and ".join(labels) + " from your search")
        if matched_tags:
            parts.append("it includes related tags like " + ", ".join(matched_tags))
        if not parts:
            return "Recommended because its content is close to your search words."
        return "Recommended because " + " and ".join(parts) + "."

    def _rank_from_vector(self, query_vector, exclude_ids: set[int] | None = None, top_n: int = 10) -> list[tuple[int, float]]:
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        exclude_ids = exclude_ids or set()
        ranked: list[tuple[int, float]] = []
        for index in np.argsort(similarities)[::-1]:
            movie_id = int(self.movies.iloc[index]["movieId"])
            if movie_id in exclude_ids:
                continue
            score = float(similarities[index])
            if score <= 0:
                continue
            ranked.append((index, score))
            if len(ranked) >= top_n:
                break
        return ranked

    def recommend_similar(self, title: str, top_n: int = 10) -> tuple[pd.Series | None, list[dict]]:
        movie = self.find_movie(title)
        if movie is None:
            return None, []

        index = self.id_to_index[int(movie["movieId"])]
        query_vector = self.tfidf_matrix[index]
        source_genres = set(movie["genre_list"])
        source_tags = set(str(movie["tags_text"]).split()) if movie["tags_text"] else set()

        ranked = self._rank_from_vector(query_vector, exclude_ids={int(movie["movieId"])}, top_n=top_n)
        results = []
        for row_index, score in ranked:
            candidate = self.movies.iloc[row_index]
            reason = self._explain_overlap(source_genres, source_tags, candidate)
            results.append(self._row_to_result(candidate, score, reason))
        return movie, results

    def _expand_query(self, query: str) -> tuple[str, set[str]]:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if token not in {"movie", "movies", "film", "films", "kind", "please", "watch", "something"}
        ]
        extras: list[str] = []
        genres: set[str] = set()
        for token in tokens:
            if token in QUERY_SYNONYMS:
                extras.append(QUERY_SYNONYMS[token])
            genres.update(WORD_TO_GENRES.get(token, []))
        expanded = (" ".join(tokens) + " " + " ".join(extras)).strip()
        return expanded, genres

    def recommend_from_text(self, query: str, top_n: int = 10) -> list[dict]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        expanded, preferred = self._expand_query(cleaned)
        query_vector = self.vectorizer.transform([expanded])
        ranked = self._rank_from_vector(query_vector, top_n=max(top_n * 8, 40))
        results = []
        for row_index, score in ranked:
            candidate = self.movies.iloc[row_index]
            reason = self._explain_query(cleaned, candidate)
            item = self._row_to_result(candidate, score, reason)
            if preferred.intersection(item["genre_list"]):
                item["similarity"] = min(100.0, round(item["similarity"] + 10, 1))
            results.append(item)
        results.sort(key=lambda item: (item["similarity"], item["avg_rating"] or 0), reverse=True)
        if preferred:
            matched = [item for item in results if preferred.intersection(item["genre_list"])]
            others = [item for item in results if not preferred.intersection(item["genre_list"])]
            if len(preferred) >= 2:
                both = [item for item in matched if preferred.issubset(set(item["genre_list"]))]
                partial = [item for item in matched if not preferred.issubset(set(item["genre_list"]))]
                if both:
                    matched = both + partial
            if len(matched) >= min(top_n, 5):
                results = matched + others
        return results[:top_n]

    def recommend_by_mood(self, mood: str, top_n: int = 10) -> list[dict]:
        profile = MOOD_PROFILES.get(mood)
        if profile is None:
            return []
        results = self.recommend_from_text(profile["query"], top_n=top_n * 3)
        preferred = profile["genres"]
        boosted = []
        for item in results:
            bonus = 8 if preferred.intersection(item["genre_list"]) else 0
            item = dict(item)
            item["similarity"] = min(100.0, round(item["similarity"] + bonus, 1))
            if preferred.intersection(item["genre_list"]):
                shared = sorted(preferred.intersection(item["genre_list"]))
                item["reason"] = f"Recommended because this {mood.lower()} pick shares " + " and ".join(shared) + " characteristics."
            boosted.append(item)
        boosted.sort(key=lambda item: (item["similarity"], item["avg_rating"] or 0), reverse=True)
        return boosted[:top_n]

    def recommend_personalized(self, titles: list[str], top_n: int = 10) -> tuple[list[pd.Series], list[dict]]:
        selected_rows = []
        vectors = []
        exclude_ids: set[int] = set()
        source_genres: set[str] = set()
        source_tags: set[str] = set()

        for title in titles:
            movie = self.find_movie(title)
            if movie is None:
                continue
            selected_rows.append(movie)
            exclude_ids.add(int(movie["movieId"]))
            source_genres.update(movie["genre_list"])
            if movie["tags_text"]:
                source_tags.update(str(movie["tags_text"]).split())
            vectors.append(self.tfidf_matrix[self.id_to_index[int(movie["movieId"])]])

        if not vectors:
            return [], []

        combined = vectors[0]
        for extra in vectors[1:]:
            combined = combined + extra
        combined = combined / len(vectors)

        ranked = self._rank_from_vector(combined, exclude_ids=exclude_ids, top_n=top_n)
        liked = ", ".join(row["title"] for row in selected_rows)
        results = []
        for row_index, score in ranked:
            candidate = self.movies.iloc[row_index]
            reason = self._explain_overlap(source_genres, source_tags, candidate)
            item = self._row_to_result(candidate, score, reason)
            item["because"] = liked
            results.append(item)
        return selected_rows, results

    def weighted_score(self, row: pd.Series, min_votes: int) -> float:
        votes = int(row["rating_count"])
        rating = float(row["avg_rating"]) if not pd.isna(row["avg_rating"]) else 0.0
        if votes == 0:
            return 0.0
        mean = self.global_mean_rating
        return (votes / (votes + min_votes)) * rating + (min_votes / (votes + min_votes)) * mean

    def top_rated(self, min_votes: int = 50, top_n: int = 20) -> list[dict]:
        eligible = self.movies[self.movies["rating_count"] >= min_votes].copy()
        if eligible.empty:
            eligible = self.movies[self.movies["rating_count"] > 0].copy()
        if eligible.empty:
            return []
        eligible["score"] = eligible.apply(lambda row: self.weighted_score(row, min_votes), axis=1)
        eligible = eligible.sort_values(["score", "rating_count"], ascending=False).head(top_n)
        results = []
        for _, row in eligible.iterrows():
            reason = (
                f"Included because it has {int(row['rating_count'])} ratings "
                f"and a reliable average of {row['avg_rating']:.2f}."
            )
            item = self._row_to_result(row, 0.0, reason)
            item["similarity"] = None
            results.append(item)
        return results

    def movies_by_genre(self, genre: str, top_n: int = 20) -> list[dict]:
        if not genre:
            return []
        subset = self.movies[self.movies["genre_list"].apply(lambda items: genre in items)].copy()
        if subset.empty:
            return []
        min_votes = 20
        subset["score"] = subset.apply(lambda row: self.weighted_score(row, min_votes), axis=1)
        subset = subset.sort_values(["score", "rating_count"], ascending=False).head(top_n)
        results = []
        for _, row in subset.iterrows():
            reason = f"Listed because it belongs to the {genre} genre."
            item = self._row_to_result(row, 0.0, reason)
            item["similarity"] = None
            results.append(item)
        return results
    def recommend_semantic(self, query: str, top_n: int = 10) -> list[dict]:
        results = self.semantic_engine.recommend_from_text(query, top_n=top_n)

        output = []

        for _, row in results.iterrows():
            item = self._row_to_result(
                row,
                float(row["semantic_score"]),
                "Recommended using semantic meaning of your search."
            )
            output.append(item)

        return output
    
