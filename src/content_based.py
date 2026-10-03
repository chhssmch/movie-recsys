import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_movies, GENRES

def build_content_matrix(movies: pd.DataFrame):
    def genres_to_str(row):
        return " ".join(g for g in GENRES if row[g] == 1)

    movies = movies.copy()
    movies["genres_str"] = movies.apply(genres_to_str, axis=1)

    # TF-IDF(t, d) = TF(t, d) × IDF(t) (высчитывание веса жанра относительно ОДНОГО фильма)
    # TF(t, d) = сколько раз t встречается в d (Term Frequency)
    # IDF(t) = ln( (1 + N) / (1 + df(t)) ) + 1 — насколько слово редкое во всей коллекции (Inverse Document Frequency)
    tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z\-']+")
    matrix = tfidf.fit_transform(movies["genres_str"]) # итог - разреженная матрица

    return movies, matrix, tfidf


def get_similar_movies(movie_title: str, movies: pd.DataFrame, matrix,
                       top_k: int = 10) -> pd.DataFrame:
    mask = movies["title"].str.contains(movie_title, case=False, regex=False)
    if not mask.any():
        raise ValueError(f"Фильм '{movie_title}' не найден")

    idx = movies[mask].index[0]
    movie_row = movies.loc[idx]

    # косинусная похожесть измеряет угол между двумя векторами
    # cos(a, b) = (a · b) / (|a| × |b|)
    sim_scores = cosine_similarity(matrix[idx], matrix).flatten()
    sim_scores[idx] = -1  # исключаем сам фильм

    # топ-K индексов похожих фильмов с исключением
    top_indices = np.argsort(sim_scores)[::-1][:top_k]

    result = movies.iloc[top_indices][["movie_id", "title"]].copy()
    result["similarity"] = sim_scores[top_indices].round(3)
    return result.reset_index(drop=True)


def recommend_for_user(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame,
                       matrix, top_k: int = 10, liked_threshold: float = 4.0) -> pd.DataFrame:
    user_ratings = ratings[ratings["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= liked_threshold]["movie_id"].tolist()
    watched = set(user_ratings["movie_id"])

    if not liked:
        return pd.DataFrame(columns=["movie_id", "title", "score"]) # пустой список, чтобы логика не упала

    liked_indices = movies[movies["movie_id"].isin(liked)].index.tolist()
    if not liked_indices:
        return pd.DataFrame(columns=["movie_id", "title", "score"]) # пустой список, чтобы логика не упала

    # для каждого любимого фильма считаем его похожесть на все остальные
    sim_sum = np.zeros(movies.shape[0])
    for idx in liked_indices:
        sim_sum += cosine_similarity(matrix[idx], matrix).flatten()
    avg_sim = sim_sum / len(liked_indices) # средний вектор похожести

    mask_watched = movies["movie_id"].isin(watched).values
    avg_sim[mask_watched] = -1 # исключаем просмотренное

    top_indices = np.argsort(avg_sim)[::-1][:top_k]

    result = movies.iloc[top_indices][["movie_id", "title"]].copy()
    result["score"] = avg_sim[top_indices].round(3)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import load_ratings

    movies = load_movies()
    ratings = load_ratings()

    movies, matrix, tfidf = build_content_matrix(movies)
    print(f"Матрица признаков: {matrix.shape}")

    print("\nПОХОЖИЕ НА 'Toy Story'")
    print(get_similar_movies("Toy Story", movies, matrix, top_k=10).to_string(index=False))

    print("\nПОХОЖИЕ НА 'Star Wars'")
    print(get_similar_movies("Star Wars", movies, matrix, top_k=10).to_string(index=False))

    print("\nРЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ 1 (content-based)")
    print(recommend_for_user(1, ratings, movies, matrix, top_k=10).to_string(index=False))
    