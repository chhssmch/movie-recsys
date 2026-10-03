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


def build_user_similarity(matrix: csr_matrix) -> np.ndarray:
    # косинусная похожесть между пользователями (попарно), результат - матрица (943, 943)
    sim = cosine_similarity(matrix, dense_output=True)
    np.fill_diagonal(sim, 0) # обнуляем диагональ
    return sim


def recommend_for_user(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame,
                       user_sim: np.ndarray, user_to_idx: dict,
                       item_to_idx: dict, idx_to_item: dict,
                       top_k: int = 10, n_neighbors: int = 30) -> pd.DataFrame:

    if user_id not in user_to_idx:
        return pd.DataFrame(columns=["movie_id", "title", "score"])

    uidx = user_to_idx[user_id]
    watched = set(ratings[ratings["user_id"] == user_id]["movie_id"])

    sim_scores = user_sim[uidx] # вектор с похожими пользователями
    neighbor_indices = np.argsort(sim_scores)[::-1][:n_neighbors] # индексы 30 самых похожих

    neighbor_indices = [i for i in neighbor_indices if sim_scores[i] > 0]
    if not neighbor_indices:
        return pd.DataFrame(columns=["movie_id", "title", "score"])

    item_scores = {}
    item_sim_sums = {}

    idx_to_user = {v: k for k, v in user_to_idx.items()}
    neighbor_ids = [idx_to_user[i] for i in neighbor_indices]

    neighbor_ratings = ratings[ratings["user_id"].isin(neighbor_ids)]
    neighbor_ratings = neighbor_ratings[neighbor_ratings["rating"] >= 4]
    neighbor_ratings = neighbor_ratings[~neighbor_ratings["movie_id"].isin(watched)] # все оценки соседей в одной таблице

    for _, row in neighbor_ratings.iterrows():
        mid = row["movie_id"]
        nid = row["user_id"]
        w = sim_scores[user_to_idx[nid]]
        item_scores[mid] = item_scores.get(mid, 0) + w * row["rating"] # вектор похожести соседа на user x его оценка
        item_sim_sums[mid] = item_sim_sums.get(mid, 0) + w

    if not item_scores:
        return pd.DataFrame(columns=["movie_id", "title", "score"])

    # score(X) = Σ (похожесть(сосед, user) × оценка(соседа)) / Σ похожесть(сосед, user)
    final = {mid: item_scores[mid] / item_sim_sums[mid] for mid in item_scores}
    top_items = sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_k]

    rows = []
    for mid, sc in top_items:
        t = movies[movies["movie_id"] == mid]["title"]
        title = t.values[0] if len(t) else "?"
        rows.append({"movie_id": mid, "title": title, "score": round(float(sc), 3)})
    return pd.DataFrame(rows)


def get_similar_users(user_id: int, user_sim: np.ndarray,
                      user_to_idx: dict, top_k: int = 10) -> pd.DataFrame:
    if user_id not in user_to_idx:
        raise ValueError(f"Пользователь {user_id} не найден")

    uidx = user_to_idx[user_id]
    scores = user_sim[uidx]
    top_indices = np.argsort(scores)[::-1][:top_k]

    idx_to_user = {v: k for k, v in user_to_idx.items()}
    return pd.DataFrame({
        "user_id": [idx_to_user[i] for i in top_indices],
        "similarity": [round(float(scores[i]), 3) for i in top_indices],
    })


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()

    print("Строим матрицу user × item...")
    matrix, user_to_idx, item_to_idx, idx_to_item = build_user_item_matrix(ratings)
    print(f"Матрица: {matrix.shape}")

    print("\nСчитаем похожесть пользователей...")
    user_sim = build_user_similarity(matrix)
    print(f"Матрица похожести: {user_sim.shape}")

    print("\nПОЛЬЗОВАТЕЛИ, ПОХОЖИЕ НА 1")
    print(get_similar_users(1, user_sim, user_to_idx, top_k=10).to_string(index=False))

    print("\nРЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ 1 (user-based CF)")
    print(recommend_for_user(1, ratings, movies, user_sim,
                             user_to_idx, item_to_idx, idx_to_item,
                             top_k=10, n_neighbors=30).to_string(index=False))