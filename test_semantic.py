from data_loader import load_dataset
from semantic_engine import SemanticMovieEngine

print("Loading movie dataset...")

data = load_dataset()
movies = data["movies"]

engine = SemanticMovieEngine(movies)

query = "funny adventure movie with friendship"

results = engine.recommend_from_text(query, top_n=10)

print("\nAI SEMANTIC RECOMMENDATIONS")
print("---------------------------")

for _, movie in results.iterrows():
    print(movie["title"])