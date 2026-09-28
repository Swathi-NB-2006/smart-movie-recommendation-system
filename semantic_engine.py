from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticMovieEngine:

    def __init__(self, movies):
        self.movies = movies.reset_index(drop=True)

        print("Loading MiniLM model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Use the movie content already created by your project
        movie_texts = self.movies["content"].fillna("").tolist()

        print("Creating movie embeddings...")
        self.movie_embeddings = self.model.encode(
            movie_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def recommend_from_text(self, query, top_n=10):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        similarities = np.dot(
            self.movie_embeddings,
            query_embedding
        )

        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = self.movies.iloc[top_indices].copy()

        results["semantic_score"] = similarities[top_indices]

        return results