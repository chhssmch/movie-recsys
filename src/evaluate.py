import numpy as np
import pandas as pd
from src.data_loader import load_ratings, load_movies, train_test_split_by_user

# метрики
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for m in rec_k if m in relevant)
    return hits / k # сколько из рекомендаций попали в целом


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Какую долю релевантных мы нашли в top-K."""
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for m in rec_k if m in relevant)
    return hits / len(relevant) # сколько было найдено именно из любимых 


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    # DCG - все релевантные стоят в начале списка
    # Normalized Discounted Cumulative Gain - нормированное значение от 0 до 1
    rec_k = recommended[:k]
    dcg = 0.0
    # DCG = Σ (1, если релевантный на позиции i) / log2(i + 2)
    for i, m in enumerate(rec_k):
        if m in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 тк log2(1)=0 (избегаем деления на ноль)

    n_rel = min(len(relevant), k)
    # IDCG - идеальное значение для оценки метрики
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0

# обертки
def make_popularity_fn(ratings: pd.DataFrame):
    stats = ratings.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
    stats = stats[stats["count"] >= 50].sort_values("mean", ascending=False)
    popular_ids = stats["movie_id"].tolist()

    def recommend(user_id, k, exclude):
        result = [m for m in popular_ids if m not in exclude][:k]
        return result
    return recommend


def make_content_fn(movies: pd.DataFrame, ratings: pd.DataFrame):
    from src.content_based import build_content_matrix
    from sklearn.metrics.pairwise import cosine_similarity

    movies_c, matrix, _ = build_content_matrix(movies)
    movie_id_to_idx = {m: i for i, m in enumerate(movies_c["movie_id"])}

    def recommend(user_id, k, exclude):
        user_rated = ratings[(ratings["user_id"] == user_id) & (ratings["rating"] >= 4)]
        liked = user_rated["movie_id"].tolist()
        liked_idx = [movie_id_to_idx[m] for m in liked if m in movie_id_to_idx]
        if not liked_idx:
            return []

        sim_sum = np.zeros(movies_c.shape[0])
        for idx in liked_idx:
            sim_sum += cosine_similarity(matrix[idx], matrix).flatten()
        avg_sim = sim_sum / len(liked_idx)

        for m in exclude:
            if m in movie_id_to_idx:
                avg_sim[movie_id_to_idx[m]] = -1

        top_idx = np.argsort(avg_sim)[::-1][:k]
        return [movies_c.iloc[i]["movie_id"] for i in top_idx]
    return recommend


def make_item_cf_fn(ratings: pd.DataFrame):
    from src.item_based_cf import build_user_item_matrix, build_item_similarity

    matrix, user_to_idx, item_to_idx, idx_to_item = build_user_item_matrix(ratings)
    sim = build_item_similarity(matrix)

    user_history = ratings.groupby("user_id").apply(
        lambda df: list(zip(df["movie_id"], df["rating"]))
    ).to_dict()

    def recommend(user_id, k, exclude):
        if user_id not in user_history:
            return []
        history = [(m, r) for m, r in user_history[user_id] if r >= 4 and m in item_to_idx]
        if not history:
            return []

        scores = np.zeros(sim.shape[0])
        sim_sums = np.zeros(sim.shape[0])
        for m, r in history:
            idx = item_to_idx[m]
            scores += sim[idx] * r
            sim_sums += sim[idx]
        sim_sums[sim_sums == 0] = 1
        scores = scores / sim_sums

        for m in exclude:
            if m in item_to_idx:
                scores[item_to_idx[m]] = -1

        top_idx = np.argsort(scores)[::-1][:k]
        return [idx_to_item[i] for i in top_idx]
    return recommend


def make_user_cf_fn(ratings: pd.DataFrame, n_neighbors: int = 30):
    from src.user_based_cf import build_user_item_matrix, build_user_similarity

    matrix, user_to_idx, item_to_idx, idx_to_item = build_user_item_matrix(ratings)
    sim = build_user_similarity(matrix)
    idx_to_user = {v: k for k, v in user_to_idx.items()}

    user_history = ratings.groupby("user_id").apply(
        lambda df: list(zip(df["movie_id"], df["rating"]))
    ).to_dict()

    def recommend(user_id, k, exclude):
        if user_id not in user_to_idx:
            return []
        uidx = user_to_idx[user_id]

        neighbor_idx = np.argsort(sim[uidx])[::-1][:n_neighbors]
        neighbor_idx = [i for i in neighbor_idx if sim[uidx][i] > 0]
        if not neighbor_idx:
            return []

        neighbor_ids = [idx_to_user[i] for i in neighbor_idx]
        scores = {}
        sim_sums = {}
        for nid in neighbor_ids:
            w = sim[uidx][user_to_idx[nid]]
            for m, r in user_history.get(nid, []):
                if r < 4 or m in exclude:
                    continue
                scores[m] = scores.get(m, 0) + w * r
                sim_sums[m] = sim_sums.get(m, 0) + w

        final = {m: scores[m] / sim_sums[m] for m in scores}
        top = sorted(final.items(), key=lambda x: x[1], reverse=True)[:k]
        return [m for m, _ in top]
    return recommend

# оценка, возвращающая средние Precision@K, Recall@K, NDCG@K и coverage
def evaluate_model(recommend_fn, test_ratings: pd.DataFrame,
                   train_ratings: pd.DataFrame, k: int = 10,
                   max_users: int = None) -> dict:

    test_users = test_ratings["user_id"].unique()
    if max_users:
        test_users = test_users[:max_users]

    precisions, recalls, ndcgs = [], [], []
    all_recommended = set()

    for uid in test_users:
        relevant = set(test_ratings[
            (test_ratings["user_id"] == uid) & (test_ratings["rating"] >= 4)
        ]["movie_id"])

        if not relevant:
            continue

        exclude = set(train_ratings[train_ratings["user_id"] == uid]["movie_id"])
        recs = recommend_fn(uid, k, exclude)

        if not recs:
            continue

        all_recommended.update(recs)
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    n_items = test_ratings["movie_id"].nunique() + train_ratings["movie_id"].nunique()
    return {
        f"Precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"Recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "Coverage": round(len(all_recommended) / n_items, 4) if n_items else 0.0,
        "n_users_evaluated": len(precisions),
    }


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()

    train, test = train_test_split_by_user(ratings, test_size=0.2)
    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Пользователей в тесте: {test['user_id'].nunique()}")

    models = {
        "Popularity": make_popularity_fn(train),
        "Content-based": make_content_fn(movies, train),
        "Item-CF": make_item_cf_fn(train),
        "User-CF": make_user_cf_fn(train),
    }

    results = []
    for name, fn in models.items():
        print(f"\nОцениваем {name}...")
        metrics = evaluate_model(fn, test, train, k=10, max_users=None)
        metrics["model"] = name
        results.append(metrics)
        print(f"  {metrics}")

    df = pd.DataFrame(results).set_index("model")
    df = df[["Precision@10", "Recall@10", "NDCG@10", "Coverage", "n_users_evaluated"]]
    print("\nИТОГОВАЯ ТАБЛИЦА")
    print(df.to_string())

    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "reports" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nСохранено в {out}")