import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import (
    load_ratings, load_movies, load_users,
    save_processed, train_test_split_by_user,
)
from src.evaluate import (
    evaluate_model,
    make_popularity_fn,
    make_content_fn,
    make_item_cf_fn,
    make_user_cf_fn,
)


def main():
    t_start = time.time()
    print("MOVIE RECSYS — ПОЛНЫЙ ПАЙПЛАЙН")

    print("\n[1/4] Загрузка данных...")
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    print(f" Ratings: {ratings.shape}")
    print(f" Movies : {movies.shape}")
    print(f" Users  : {users.shape}")

    save_processed()

    print("\n[2/4] Train/test split (80/20 по пользователям)...")
    train, test = train_test_split_by_user(ratings, test_size=0.2)
    print(f" Train: {len(train)}")
    print(f" Test : {len(test)}")
    print(f" Пользователей в тесте: {test['user_id'].nunique()}")

    print("\n[3/4] Инициализация моделей...")
    models = {}

    print(" Popularity (baseline)...")
    models["Popularity"] = make_popularity_fn(train)

    print(" Content-based...")
    models["Content-based"] = make_content_fn(movies, train)

    print(" Item-based CF...")
    models["Item-CF"] = make_item_cf_fn(train)

    print(" User-based CF...")
    models["User-CF"] = make_user_cf_fn(train, n_neighbors=30)

    print("\n[4/4] Оценка моделей...")
    K = 10
    MAX_USERS = None

    results = []
    for name, fn in models.items():
        print(f"\n {name}...")
        t0 = time.time()
        metrics = evaluate_model(fn, test, train, k=K, max_users=MAX_USERS)
        metrics["model"] = name
        metrics["time_sec"] = round(time.time() - t0, 2)
        results.append(metrics)
        print(f"    Precision@{K}={metrics[f'Precision@{K}']:.4f}  "
              f"Recall@{K}={metrics[f'Recall@{K}']:.4f}  "
              f"NDCG@{K}={metrics[f'NDCG@{K}']:.4f}  "
              f"Coverage={metrics['Coverage']:.4f}  "
              f"({metrics['time_sec']}s)")

    df = pd.DataFrame(results).set_index("model")
    cols = [f"Precision@{K}", f"Recall@{K}", f"NDCG@{K}",
            "Coverage", "n_users_evaluated", "time_sec"]
    df = df[cols]

    out = ROOT / "reports" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)

    print("\nИТОГОВАЯ ТАБЛИЦА")
    print(df.to_string())

    print(f"\nСохранено: {out}")

    best = df[f"NDCG@{K}"].idxmax()
    print(f"\nЛучшая модель по NDCG@{K}: {best} "
          f"({df.loc[best, f'NDCG@{K}']:.4f})")

    fastest = df["time_sec"].idxmin()
    print(f"Самая быстрая: {fastest} ({df.loc[fastest, 'time_sec']}s)")

    best_cov = df["Coverage"].idxmax()
    print(f"Максимальный Coverage: {best_cov} "
          f"({df.loc[best_cov, 'Coverage']:.4f})")

    print(f"\nОбщее время: {round(time.time() - t_start, 1)}s")
    print("ГОТОВО. Смотри reports/metrics.csv и docs/ для отчёта.")

if __name__ == "__main__":
    main()