🎬 Smart Movie Recommendation System Using Machine Learning

A content-based movie recommendation system that helps users discover movies based on movie similarity, natural-language descriptions, moods, genres, and personalized selections.

The project uses TF-IDF and cosine similarity for traditional content-based recommendations and includes a semantic recommendation engine using the MiniLM Sentence Transformer model to understand the meaning of a user's search query.

✨ Features

- 🎯 Recommend movies similar to a selected movie
- 🧠 Semantic search using MiniLM embeddings
- 🔍 Search for movies using natural-language descriptions
- 😊 Mood-based movie recommendations
- ⭐ Personalized recommendations based on selected movies
- 🏆 Top-rated movie section
- 🎭 Browse movies by genre
- 🎞️ Watchlist support
- 📊 Movie and rating statistics
- 🌐 Interactive Streamlit web interface
- 🧪 Semantic recommendation testing

🧠 Machine Learning Approach

1. Movie Dataset

The system works with movie information containing titles, genres, tags, and rating information.

2. Content Processing

Movie information is combined into content text that can be processed by the recommendation system.

3. TF-IDF

TF-IDF converts movie content into numerical feature vectors.

4. Cosine Similarity

Cosine similarity measures how similar two movie-content vectors are.

5. Semantic Recommendation

The project also uses the "all-MiniLM-L6-v2" Sentence Transformer model.

Instead of relying only on matching individual words, semantic embeddings represent the meaning of movie content and the user's query.

6. Recommendation

Movies are ranked according to their similarity to the selected movie or search description.

🔄 Project Flow

Movie Dataset
↓
Content / Features
↓
TF-IDF & Semantic Embeddings
↓
Similarity Calculation
↓
Movie Ranking
↓
Recommendations

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- Streamlit
- Plotly

📁 Project Structure

smart-movie-recommendation/
│
├── app.py
├── data_loader.py
├── recommendation_engine.py
├── semantic_engine.py
├── test_semantic.py
├── requirements.txt
├── README.md
└── .gitignore

🚀 How to Run

1. Clone the repository

git clone <your-repository-url>
cd smart-movie-recommendation

2. Install the required packages

pip install -r requirements.txt

3. Start the Streamlit application

python -m streamlit run app.py

4. Open the application

Streamlit will provide a local URL, usually:

http://localhost:8501

Open it in your browser.

🔎 Example Semantic Search

Users can enter natural-language queries such as:

a funny adventure movie

or

a romantic emotional love story

The semantic engine converts the query into an embedding and compares it with movie embeddings to find relevant recommendations.

🧪 Testing

The project includes:

test_semantic.py

for testing the semantic recommendation functionality.

🎓 Project Information

Project: Smart Movie Recommendation System Using Machine Learning
Branch: Electronics and Communication Engineering (ECE)
Academic Year: 2026–27

📌 Future Improvements

- Movie poster and trailer integration
- User login and persistent profiles
- Improved recommendation using user feedback
- Larger movie datasets
- Deployment as a public web application
- Further optimization for edge/AI hardware platforms

👩‍💻 Author

Swathi N.B
ECE 3rd Year Engineering Student

---

⭐ If you find this project interesting, feel free to explore the code and recommendation engine.
