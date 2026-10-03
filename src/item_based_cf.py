import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_ratings, load_movies


def build_user_item_matrix(ratings: pd.DataFrame):
    users = ratings["user_id"].unique()
    items = ratings["movie_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {m: i for i, m in enumerate(items)}
    idx_to_item = {i: m for m, i in item_to_idx.items()}

    rows = ratings["user_id"].map(user_to_idx).values
    cols = ratings["movie_id"].map(item_to_idx).values
    vals = ratings["rating"].values.astype(np.float32)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items))) # разреженная матрица (пользователи x фильмы)
    return matrix, user_to_idx, item_to_idx, idx_to_item


def build_item_similarity(matrix: csr_matrix) -> np.ndarray:
    item_user = matrix.T # меняем строки и столбцы местами (транспонирование)
    # cos(a, b) = (a · b) / (|a| × |b|)
    # косинусная похожесть между строками (фильмами попарно), результат - матрица (1682, 1682)
    sim = cosine_similarity(item_user, dense_output=True)
    np.fill_diagonal(sim, 0) # похожесть фильма самого с собой заменяем на 0
    return sim


def recommend_for_user(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame,
                       sim_matrix: np.ndarray, user_to_idx: dict,
                       item_to_idx: dict, idx_to_item: dict,
                       top_k: int = 10, min_rating: float = 4.0) -> pd.DataFrame:
    if user_id not in user_to_idx:
        return pd.DataFrame(columns=["movie_id", "title", "score"])

    user_ratings = ratings[ratings["user_id"] == user_id]
    watched = set(user_ratings["movie_id"])
    liked = user_ratings[user_ratings["rating"] >= min_rating]
    
    if liked.empty:
        return pd.DataFrame(columns=["movie_id", "title", "score"])

    liked_idx = [item_to_idx[m] for m in liked["movie_id"] if m in item_to_idx]
    liked_ratings = liked["rating"].values

    # scores — накопленный балл
    # sim_sums — накопленная сумма похожестей
    scores = np.zeros(sim_matrix.shape[0])
    sim_sums = np.zeros(sim_matrix.shape[0])

    for idx, r in zip(liked_idx, liked_ratings):
        scores += sim_matrix[idx] * r # вектор похожести ОДНОГО фильма на другие x его оценка
        sim_sums += sim_matrix[idx]

    sim_sums[sim_sums == 0] = 1 # нормирование, чтобы избежать деления на ноль

    # score(X) = Σ (похожесть(X, любимый_i) × оценка_i) / Σ похожесть(X, любимый_i)
    scores = scores / sim_sums # получаем взвешенное среднее

    for m in watched:
        if m in item_to_idx:
            scores[item_to_idx[m]] = -1 # исключение просмотренного

    top_indices = np.argsort(scores)[::-1][:top_k]

    result_rows = []
    for idx in top_indices:
        movie_id = idx_to_item[idx]
        title_row = movies[movies["movie_id"] == movie_id]["title"]
        title = title_row.values[0] if len(title_row) else "?"
        result_rows.append({"movie_id": movie_id, "title": title,
                            "score": round(float(scores[idx]), 3)})

    return pd.DataFrame(result_rows)


def get_similar_items(movie_id: int, sim_matrix: np.ndarray,
                      idx_to_item: dict, item_to_idx: dict,
                      movies: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if movie_id not in item_to_idx:
        raise ValueError(f"Фильм {movie_id} не найден в матрице")

    idx = item_to_idx[movie_id]
    scores = sim_matrix[idx] # просто смотрим на вектор похожести
    top_indices = np.argsort(scores)[::-1][:top_k]

    rows = []
    for i in top_indices:
        mid = idx_to_item[i]
        title_row = movies[movies["movie_id"] == mid]["title"]
        title = title_row.values[0] if len(title_row) else "?"
        rows.append({"movie_id": mid, "title": title,
                     "similarity": round(float(scores[i]), 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()

    print("Строим матрицу user × item...")
    matrix, user_to_idx, item_to_idx, idx_to_item = build_user_item_matrix(ratings)
    print(f"Матрица: {matrix.shape}, заполнено: {matrix.nnz / (matrix.shape[0]*matrix.shape[1]):.2%} (потому что каждый пользователь смотрел лишь малую часть всех фильмов)")

    print("\nСчитаем похожесть фильмов...")
    sim_matrix = build_item_similarity(matrix)
    print(f"Матрица похожести: {sim_matrix.shape}")

    print("\nФИЛЬМЫ, ПОХОЖИЕ НА 'Toy Story' (по поведению пользователей)")
    print(get_similar_items(1, sim_matrix, idx_to_item, item_to_idx,
                            movies, top_k=10).to_string(index=False))

    print("\nРЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ 1 (item-based CF)")
    print(recommend_for_user(1, ratings, movies, sim_matrix,
                             user_to_idx, item_to_idx, idx_to_item,
                             top_k=10).to_string(index=False))