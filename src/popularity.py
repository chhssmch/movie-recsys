import pandas as pd
from src.data_loader import load_ratings, load_movies

def get_top_popular(ratings: pd.DataFrame, movies: pd.DataFrame,
                    top_k: int = 10, min_votes: int = 50) -> pd.DataFrame:
    stats = ratings.groupby("movie_id").agg(
        avg_rating=("rating", "mean"),
        n_votes=("rating", "count"),
    ).reset_index()

    stats = stats[stats["n_votes"] >= min_votes] # исключение фильмов с малым числом оценок
    stats = stats.sort_values("avg_rating", ascending=False).head(top_k)

    return stats.merge(movies[["movie_id", "title"]], on="movie_id")


def get_top_by_genre(ratings: pd.DataFrame, movies: pd.DataFrame,
                     genre: str, top_k: int = 10, min_votes: int = 30) -> pd.DataFrame:
    if genre not in movies.columns:
        raise ValueError(f"Жанр '{genre}' не найден. Доступные: {list(movies.columns[5:])}")

    genre_movies = movies[movies[genre] == 1][["movie_id", "title"]]
    ratings_genre = ratings[ratings["movie_id"].isin(genre_movies["movie_id"])]

    stats = ratings_genre.groupby("movie_id").agg(
        avg_rating=("rating", "mean"),
        n_votes=("rating", "count"),
    ).reset_index()

    stats = stats[stats["n_votes"] >= min_votes]
    stats = stats.sort_values("avg_rating", ascending=False).head(top_k)

    return stats.merge(genre_movies, on="movie_id")


def get_top_by_decade(ratings: pd.DataFrame, movies: pd.DataFrame,
                      decade: int, top_k: int = 10, min_votes: int = 30) -> pd.DataFrame:
    movies = movies.copy() # тк добавляем год, чтобы не затронуть оригинальный DataFrame
    movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    decade_movies = movies[(movies["year"] >= decade) & (movies["year"] < decade + 10)]
    decade_movies = decade_movies[["movie_id", "title", "year"]]

    ratings_dec = ratings[ratings["movie_id"].isin(decade_movies["movie_id"])]

    stats = ratings_dec.groupby("movie_id").agg(
        avg_rating=("rating", "mean"),
        n_votes=("rating", "count"),
    ).reset_index()

    stats = stats[stats["n_votes"] >= min_votes]
    stats = stats.sort_values("avg_rating", ascending=False).head(top_k)

    return stats.merge(decade_movies, on="movie_id")

# baseline для сравнения
def recommend_for_user(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame,
                       top_k: int = 10) -> pd.DataFrame:
    watched = set(ratings[ratings["user_id"] == user_id]["movie_id"])

    stats = ratings.groupby("movie_id").agg(
        avg_rating=("rating", "mean"),
        n_votes=("rating", "count"),
    ).reset_index()
    stats = stats[~stats["movie_id"].isin(watched)] # исключаем просмотренное
    stats = stats[stats["n_votes"] >= 50]
    stats = stats.sort_values("avg_rating", ascending=False).head(top_k)

    return stats.merge(movies[["movie_id", "title"]], on="movie_id")


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()

    print("\nТОП-10 ПОПУЛЯРНЫХ ФИЛЬМОВ")
    print(get_top_popular(ratings, movies, top_k=10).to_string(index=False))

    print("\nТОП-10 КОМЕДИЙ")
    print(get_top_by_genre(ratings, movies, genre="Comedy", top_k=10).to_string(index=False))

    print("\nТОП-10 ФИЛЬМОВ 1990-Х")
    print(get_top_by_decade(ratings, movies, decade=1990, top_k=10).to_string(index=False))

    print("\nРЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ 1 (что он ещё не смотрел)")
    print(recommend_for_user(1, ratings, movies, top_k=10).to_string(index=False))