#сравнение с baseline (SVD/k-NN)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.saving import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings

# Убираем предупреждения
warnings.filterwarnings('ignore')

# Загрузка данных
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Нормализация оценок
min_rating = ratings['rating'].min()
max_rating = ratings['rating'].max()
ratings['rating_norm'] = (ratings['rating'] - min_rating) / (max_rating - min_rating)

# Фильтрация
user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()
ratings_filtered = ratings[
    (ratings['userId'].isin(user_counts[user_counts >= 5].index)) &
    (ratings['movieId'].isin(movie_counts[movie_counts >= 5].index))
    ]

# Переиндексация
user_to_idx = {uid: i for i, uid in enumerate(ratings_filtered['userId'].unique())}
movie_to_idx = {mid: i for i, mid in enumerate(ratings_filtered['movieId'].unique())}
ratings_filtered['user_idx'] = ratings_filtered['userId'].map(user_to_idx)
ratings_filtered['movie_idx'] = ratings_filtered['movieId'].map(movie_to_idx)

# Разделение данных
train, test = train_test_split(ratings_filtered, test_size=0.2, random_state=42)

# Загрузка обученной NeuMF модели
try:
    model = load_model('neuMF_optimized.keras')
    print("NeuMF модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit()


# 1. Реализация SVD "вручную" (упрощенная версия)
def manual_svd(train_data, n_factors=30, n_epochs=20, lr=0.005, reg=0.02):
    # Инициализация матриц
    n_users = train_data['user_idx'].nunique()
    n_movies = train_data['movie_idx'].nunique()

    # Матрицы факторов
    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_movies, n_factors))

    # SGD обучение
    for _ in range(n_epochs):
        for row in train_data.itertuples():
            u, i, r = row.user_idx, row.movie_idx, row.rating_norm
            err = r - np.dot(P[u], Q[i])

            # Обновление факторов
            P_u = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P_u - reg * Q[i])

    return P, Q


print("\nОбучение SVD...")
P, Q = manual_svd(train, n_factors=50, n_epochs=50, lr=0.01, reg=0.01)


# 2. Реализация User-based k-NN
def user_knn(train_data, k=40):
    # Создание user-item матрицы
    user_item = train_data.pivot(index='user_idx', columns='movie_idx', values='rating_norm').fillna(0)
    sparse_matrix = csr_matrix(user_item.values)

    # Модель k-NN
    model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)
    return model, user_item


print("\nОбучение User-based k-NN...")
knn_model, user_item_matrix = user_knn(train)


# Функции для предсказаний
def svd_predict(user_idx, movie_idx):
    try:
        # Проверяем и преобразуем индексы в целые числа
        user_idx = int(user_idx)
        movie_idx = int(movie_idx)

        # Проверяем границы индексов
        if user_idx < 0 or user_idx >= P.shape[0]:
            raise ValueError(f"Недопустимый user_idx: {user_idx}")
        if movie_idx < 0 or movie_idx >= Q.shape[0]:
            raise ValueError(f"Недопустимый movie_idx: {movie_idx}")

        return np.dot(P[user_idx], Q[movie_idx])
    except Exception as e:
        print(f"Ошибка в svd_predict: {e}")
        return (min_rating + max_rating) / 2  # Возвращаем средний рейтинг при ошибке


def knn_predict(user_idx, movie_idx, n_neighbors=20):
    try:
        # Преобразуем в целые числа
        user_idx = int(user_idx)
        movie_idx = int(movie_idx)

        # Проверяем, есть ли фильм в обучающих данных
        if movie_idx not in user_item_matrix.columns:
            return 0.5  # Нейтральное значение для новых фильмов

        # Получаем ближайших соседей
        distances, indices = knn_model.kneighbors(
            user_item_matrix.iloc[user_idx:user_idx + 1],
            n_neighbors=n_neighbors
        )

        # Собираем НЕнулевые оценки соседей
        neighbor_ratings = []
        for idx in indices[0]:
            rating = user_item_matrix.iloc[idx][movie_idx]
            if rating > 0:  # Учитываем только реальные оценки
                neighbor_ratings.append(rating)

        # Если нет оценок, возвращаем среднее по всем оценкам этого пользователя
        if not neighbor_ratings:
            user_ratings = user_item_matrix.iloc[user_idx]
            mean_rating = user_ratings[user_ratings > 0].mean()
            return mean_rating if not np.isnan(mean_rating) else 0.5

        return np.mean(neighbor_ratings)

    except Exception as e:
        print(f"Ошибка в knn_predict: {e}")
        return 0.5  # Возвращаем нейтральное значение при ошибке

# Оценка на тестовых данных
def evaluate_models(test_data):
    # Подготовка данных для батч-предсказаний
    test_users = test_data['user_idx'].values
    test_movies = test_data['movie_idx'].values
    true_ratings = test_data['rating_norm'].values

    # SVD предсказания (векторизованная версия)
    svd_preds = np.array([np.dot(P[u], Q[i]) for u, i in zip(test_users, test_movies)])

    # k-NN предсказания (оптимизированная версия)
    knn_preds = []
    batch_size = 1000  # Обрабатываем пользователей батчами
    for i in range(0, len(test_users), batch_size):
        batch_users = test_users[i:i + batch_size]
        batch_movies = test_movies[i:i + batch_size]

        # Векторизованный расчет для батча
        distances, indices = knn_model.kneighbors(
            user_item_matrix.iloc[batch_users],
            n_neighbors=10
        )

        for j, (user_idx, movie_idx) in enumerate(zip(batch_users, batch_movies)):
            neighbor_ratings = user_item_matrix.iloc[indices[j]][movie_idx]
            pred = neighbor_ratings.mean() if neighbor_ratings.notna().any() else 0.5
            knn_preds.append(pred)

    # NeuMF предсказания батчами
    neumf_preds = model.predict(
        [test_users, test_movies, test_users, test_movies],
        batch_size=1024,
        verbose=1
    ).flatten()

    # Расчет метрик
    metrics = {
        'SVD': {
            'RMSE': np.sqrt(mean_squared_error(true_ratings, svd_preds)),
            'MAE': mean_absolute_error(true_ratings, svd_preds)
        },
        'k-NN': {
            'RMSE': np.sqrt(mean_squared_error(true_ratings, knn_preds)),
            'MAE': mean_absolute_error(true_ratings, knn_preds)
        },
        'NeuMF': {
            'RMSE': np.sqrt(mean_squared_error(true_ratings, neumf_preds)),
            'MAE': mean_absolute_error(true_ratings, neumf_preds)
        }
    }
    return metrics


print("\nОценка моделей на тестовых данных...")
metrics = evaluate_models(test)

# Вывод результатов
print("\nСравнение моделей:")
for model_name, model_metrics in metrics.items():
    print(f"{model_name}: RMSE={model_metrics['RMSE']:.4f}, MAE={model_metrics['MAE']:.4f}")

# Визуализация
plt.figure(figsize=(12, 5))

# График RMSE
plt.subplot(1, 2, 1)
plt.bar(metrics.keys(), [m['RMSE'] for m in metrics.values()])
plt.title('Сравнение RMSE')
plt.ylabel('RMSE')

# График MAE
plt.subplot(1, 2, 2)
plt.bar(metrics.keys(), [m['MAE'] for m in metrics.values()])
plt.title('Сравнение MAE')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()

# Пример предсказаний
try:
    sample = test.iloc[0]
    user_idx = int(sample['user_idx'])  # Явное преобразование в int
    movie_idx = int(sample['movie_idx'])

    print("\nПример предсказаний:")
    print(f"Пользователь: {sample['userId']}")
    print(f"Фильм: {movies[movies['movieId'] == sample['movieId']]['title'].values[0]}")
    print(f"Реальный рейтинг: {sample['rating']:.2f}")

    svd_pred = svd_predict(user_idx, movie_idx)
    print(f"SVD предсказание: {svd_pred * (max_rating - min_rating) + min_rating:.2f}")

    knn_pred = knn_predict(user_idx, movie_idx)
    print(f"k-NN предсказание: {knn_pred * (max_rating - min_rating) + min_rating:.2f}")

    neumf_pred = model.predict([
        np.array([user_idx]), np.array([movie_idx]),
        np.array([user_idx]), np.array([movie_idx])
    ], verbose=0)[0][0]
    print(f"NeuMF предсказание: {neumf_pred * (max_rating - min_rating) + min_rating:.2f}")

except Exception as e:
    print(f"Ошибка при создании примеров предсказаний: {e}")




