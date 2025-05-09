import numpy as np
import pandas as pd
from keras.src.saving import load_model

# Загрузка данных
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Добавляем определение min_rating и max_rating
min_rating = ratings['rating'].min()  # Добавьте эту строку
max_rating = ratings['rating'].max()  # Добавьте эту строку

# Загрузка модели и словарей
model = load_model('neuMF_final.keras')
user_to_idx = np.load('user_to_idx.npy', allow_pickle=True).item()
movie_to_idx = np.load('movie_to_idx.npy', allow_pickle=True).item()
idx_to_movie = {v: k for k, v in movie_to_idx.items()}

# Проверка максимальных индексов
print(f"Максимальный индекс пользователей в модели: {max(user_to_idx.values())}")
print(f"Максимальный индекс фильмов в модели: {max(movie_to_idx.values())}")

for user_id in range(1,10):
    if user_id not in user_to_idx:
        print(f"Пользователь {user_id} не найден в обученных данных!")
    else:
        user_idx = user_to_idx[user_id]

        # Получаем фильмы, которые пользователь не оценивал
        rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'])
        unrated_movies = movies[~movies['movieId'].isin(rated_movies)]

        # Фильтруем только фильмы, которые есть в модели
        unrated_movies = unrated_movies[unrated_movies['movieId'].isin(movie_to_idx.keys())]

        if len(unrated_movies) == 0:
            print("Нет фильмов для рекомендации!")
        else:
            # Подготовка данных для предсказания
            movie_indices = np.array([movie_to_idx[m] for m in unrated_movies['movieId']])
            user_indices = np.array([user_idx] * len(movie_indices))

            # Предсказание
            predictions = model.predict(
                [user_indices, movie_indices, user_indices, movie_indices],
                verbose=0
            ).flatten()

            # Собираем рекомендации
            recommendations = pd.DataFrame({
                'movieId': unrated_movies['movieId'],
                'title': unrated_movies['title'],
                'pred_rating': predictions * (max_rating - min_rating) + min_rating
            })

            # Топ-5 рекомендаций
            top_5 = recommendations.sort_values('pred_rating', ascending=False).head(5)

            print(f"\nТоп-5 рекомендаций для пользователя {user_id}:")
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"{i}. {row['title']} (предсказанный рейтинг: {row['pred_rating']:.2f})")

            # Проверка работы модели
            print("\nПроверка диапазона предсказаний:")
            print(f"Минимальный рейтинг: {recommendations['pred_rating'].min():.2f}")
            print(f"Максимальный рейтинг: {recommendations['pred_rating'].max():.2f}")
            print(f"Средний рейтинг: {recommendations['pred_rating'].mean():.2f}")