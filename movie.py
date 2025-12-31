# ----------------------------------
# Movie Recommendation System
# ----------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample Movie Dataset
# -----------------------------
data = {
    "title": [
        "Avatar",
        "Titanic",
        "Avengers",
        "Iron Man",
        "The Dark Knight",
        "Interstellar",
        "Inception"
    ],
    "genre": [
        "action adventure fantasy",
        "romance drama",
        "action superhero",
        "action superhero",
        "action crime drama",
        "sci fi space",
        "sci fi thriller"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df["genre"])

# -----------------------------
# Similarity Calculation
# -----------------------------
similarity = cosine_similarity(genre_matrix)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie_name):
    if movie_name not in df["title"].values:
        return "Movie not found!"

    index = df[df["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"Movies similar to {movie_name}:\n")
    for i in scores[1:4]:
        print(df.iloc[i[0]]["title"])

# -----------------------------
# Test Recommendation
# -----------------------------
recommend("Avatar")
