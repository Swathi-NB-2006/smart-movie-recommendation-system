# Smart Movie Recommendation System

A beginner-friendly movie discovery web app built with **Python** and **Streamlit**. It recommends movies using **TF-IDF** and **cosine similarity** on movie titles, genres and tags from the **MovieLens** dataset.

This is a **content-based** recommender. It is **not** ChatGPT, and it does **not** use paid APIs or deep learning.

---

## Problem statement

People often know they want “a funny adventure movie” or “something like Toy Story”, but browsing a large catalogue is slow. This project helps a user find similar or matching movies quickly, using machine learning on movie content instead of paid recommendation APIs.

---

## Objectives

- Build a complete, runnable movie recommendation system for a college mini-project.
- Use MovieLens data without asking the user to collect files by hand.
- Recommend movies with TF-IDF and cosine similarity.
- Support natural-language search, moods, personal favourites and a watchlist.
- Explain each recommendation using genres and tags that actually exist in the data.
- Keep the project simple enough for a beginner to run from VS Code or Cursor on Windows.

---

## Features

1. **Movie recommendation** – select a movie and get the top 10 similar titles.
2. **NLP-based smart search** – type phrases such as `funny adventure movie`.
3. **Mood-based recommendations** – Happy, Romantic, Thriller, Action, Relaxing, Funny, Intelligent.
4. **Personalized recommendations** – combine up to 3 favourite movies.
5. **Explainable recommendations** – every card includes “Why this movie?”.
6. **Top rated movies** – ranked with a minimum rating-count threshold.
7. **Genre explorer** – browse one genre, ordered by a reliable rating score.
8. **Dashboard** – movie, rating and user counts in the sidebar.
9. **Watchlist** – add, remove and clear movies during the current session.

---

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

No API keys are required.

---

## Dataset

The project uses the **MovieLens Latest Small** dataset from GroupLens:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

If these files are missing, the app **downloads and extracts them automatically** into `data/ml-latest-small/`.

MovieLens is a research dataset. Please credit GroupLens / MovieLens if you present this project.

---

## How the recommendation algorithm works

```text
Movie data
    → Content creation (title + genres + tags)
    → TF-IDF vectorization
    → Cosine similarity against the selected movie or search text
    → Ranked recommendations
```

Ratings are **not** used inside TF-IDF. They are used only for:

- average rating on cards
- top-rated lists
- genre explorer ordering

The app does **not** build a full movie-to-movie similarity matrix. It calculates similarity only for the current query, which keeps memory use low.

---

## TF-IDF explanation

**TF-IDF** means Term Frequency–Inverse Document Frequency.

- **Term frequency** asks: how often does a word appear in this movie’s content?
- **Inverse document frequency** asks: is that word rare across all movies?

Common words such as `the` become less important. Distinctive words such as `animation`, `cyberpunk` or a unique tag become more important. Each movie is turned into a numeric vector.

---

## Cosine similarity explanation

**Cosine similarity** measures the angle between two vectors.

- `1.0` means the content is very similar
- `0.0` means there is little overlap

The app shows this as a **similarity percentage** on recommendation cards.

---

## Project structure

```text
smartmovie/
├── app.py                      # Streamlit user interface
├── recommendation_engine.py    # TF-IDF, cosine similarity, explanations
├── data_loader.py              # MovieLens download and preparation
├── requirements.txt
├── README.md
├── .streamlit/config.toml      # Dark charcoal + gold theme
└── data/                       # Dataset is downloaded here automatically
```

---

## Installation

In VS Code or Cursor, open a terminal in this project folder.

### 1. Create a virtual environment (recommended on Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Windows blocks the activate script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages

```powershell
pip install -r requirements.txt
```

---

## How to run

```powershell
streamlit run app.py
```

The first start may take a little longer because the MovieLens zip file is downloaded and the TF-IDF model is built. Later starts are faster because Streamlit caches that work.

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Example queries

Try these in **Discover → What kind of movie are you looking for?**

- `funny adventure movie`
- `romantic emotional movie`
- `intelligent thriller`
- `action science fiction`
- `relaxing comedy`

Also try recommending movies similar to:

- `Toy Story (1995)`
- `The Matrix (1999)`
- `Forrest Gump (1994)`

---

## Future enhancements

- Collaborative filtering using user-user or item-item ratings
- A saved watchlist (file or database) instead of session-only storage
- Poster images if a free image source is added
- Better handling of sequels and remakes
- User accounts and rating history

---

## Academic note

This project is suitable for a college mini-project demonstration. It clearly shows:

- data loading
- text preprocessing
- TF-IDF
- cosine similarity
- a usable interface
- honest explanations that stay within the dataset
