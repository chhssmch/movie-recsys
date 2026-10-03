from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings() -> pd.DataFrame:
    path = RAW_DIR / "u.data"
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )
    return df


def load_movies() -> pd.DataFrame:
    path = RAW_DIR / "u.item"
    cols = ["movie_id", "title", "release_date", "video_date", "imdb_url"] + GENRES
    df = pd.read_csv(
        path,
        sep="|",
        names=cols,
        encoding="latin-1",
        engine="python",
    )
    return df


def load_users() -> pd.DataFrame:
    path = RAW_DIR / "u.user"
    df = pd.read_csv(
        path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="python",
    )
    return df


def save_processed():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    load_ratings().to_csv(PROCESSED_DIR / "ratings.csv", index=False)
    load_movies().to_csv(PROCESSED_DIR / "movies.csv", index=False)
    load_users().to_csv(PROCESSED_DIR / "users.csv", index=False)
    print(f"Сохранено в {PROCESSED_DIR}")


def train_test_split_by_user(ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    rng = pd.Series(range(len(ratings))).sample(frac=1, random_state=seed).values
    ratings = ratings.iloc[rng].reset_index(drop=True)

    # булевая маска для деления
    test_mask = ratings.groupby("user_id").cumcount() < (
        ratings.groupby("user_id")["rating"].transform("size") * test_size
    )
    train = ratings[~test_mask].reset_index(drop=True) # False - train
    test = ratings[test_mask].reset_index(drop=True) # True - test
    return train, test


def quick_check():
    r = load_ratings()
    m = load_movies()
    u = load_users()
    print(f"ratings: {r.shape}, users: {r.user_id.nunique()}, movies: {r.movie_id.nunique()}")
    print(f"movies : {m.shape}")
    print(f"users  : {u.shape}")
    print(f"средняя оценка: {r.rating.mean():.2f}")
    return r, m, u


if __name__ == "__main__":
    quick_check()
    save_processed()