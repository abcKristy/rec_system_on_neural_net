import pandas as pd
from keras import Model

# Загрузка данных
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Вывод информации о данных
print("=== Ratings DataFrame ===")
print(ratings.head())
print("\nИнформация о ratings:")
print(ratings.info())

print("\n=== Movies DataFrame ===")
print(movies.head())
print("\nИнформация о movies:")
print(movies.info())

# Нормализация оценок (исходный диапазон: 0.5 - 5)
min_rating = ratings['rating'].min()
max_rating = ratings['rating'].max()
ratings['rating_norm'] = (ratings['rating'] - min_rating) / (max_rating - min_rating)

print("\nНормализованные рейтинги:")
print(ratings[['userId', 'movieId', 'rating', 'rating_norm']].head())

# Фильтрация пользователей
user_rating_counts = ratings['userId'].value_counts()
valid_users = user_rating_counts[user_rating_counts >= 5].index
ratings_filtered = ratings[ratings['userId'].isin(valid_users)]

# Фильтрация фильмов
movie_rating_counts = ratings['movieId'].value_counts()
valid_movies = movie_rating_counts[movie_rating_counts >= 5].index
ratings_filtered = ratings_filtered[ratings_filtered['movieId'].isin(valid_movies)]

print("\nДанные после фильтрации:")
print(f"Исходное количество записей: {len(ratings)}")
print(f"Осталось записей: {len(ratings_filtered)}")

# Создание новых последовательных ID
unique_users = ratings_filtered['userId'].unique()
unique_movies = ratings_filtered['movieId'].unique()

user_to_idx = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
movie_to_idx = {old_id: new_id for new_id, old_id in enumerate(unique_movies)}

# Применение переиндексации
ratings_filtered['user_idx'] = ratings_filtered['userId'].map(user_to_idx)
ratings_filtered['movie_idx'] = ratings_filtered['movieId'].map(movie_to_idx)

print("\nПример переиндексации:")
print(ratings_filtered[['userId', 'user_idx', 'movieId', 'movie_idx']].head())

from sklearn.model_selection import train_test_split

# Стратифицированное разделение (по userId)
train_data, test_data = train_test_split(
    ratings_filtered,
    test_size=0.2,
    stratify=ratings_filtered['user_idx'],
    random_state=42
)

print("\nРазмеры выборок:")
print(f"Train: {len(train_data)} записей")
print(f"Test: {len(test_data)} записей")

import numpy as np

# Train данные
X_train_gmf = train_data[['user_idx', 'movie_idx']].values
X_train_mlp = train_data[['user_idx', 'movie_idx']].values
y_train = train_data['rating_norm'].values

# Test данные
X_test_gmf = test_data[['user_idx', 'movie_idx']].values
X_test_mlp = test_data[['user_idx', 'movie_idx']].values
y_test = test_data['rating_norm'].values

print("\nФормы данных:")
print(f"X_train_gmf: {X_train_gmf.shape}")
print(f"X_train_mlp: {X_train_mlp.shape}")
print(f"y_train: {y_train.shape}")

# Сохранение словарей для ID
np.save('user_to_idx.npy', user_to_idx)
np.save('movie_to_idx.npy', movie_to_idx)

# Сохранение данных
np.save('X_train_gmf.npy', X_train_gmf)
np.save('X_train_mlp.npy', X_train_mlp)
np.save('y_train.npy', y_train)
np.save('X_test_gmf.npy', X_test_gmf)
np.save('X_test_mlp.npy', X_test_mlp)
np.save('y_test.npy', y_test)

print("\nДанные сохранены в .npy файлы.")