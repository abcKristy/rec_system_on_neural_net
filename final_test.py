import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.saving import load_model
from sklearn.metrics import confusion_matrix
from part2 import test
from part3 import histories

# Загрузка обученной модели
model = load_model('neuMF_final.keras')

# Загрузка данных (предполагаем, что у вас есть эти данные)
movies = pd.read_csv('movies.csv')  # Файл с информацией о фильмах
ratings = pd.read_csv('ratings.csv')  # Полный датасет оценок


# Функция для получения рекомендаций
def get_recommendations(user_id, n_recommendations=10):
    # Получаем все фильмы, которые пользователь еще не оценивал
    rated_movies = ratings[ratings['user_idx'] == user_id]['movie_idx'].unique()
    all_movies = ratings['movie_idx'].unique()
    unseen_movies = np.setdiff1d(all_movies, rated_movies)

    # Создаем пары пользователь-фильм для предсказания
    user_movie_pairs = np.array([[user_id, movie_id] for movie_id in unseen_movies])

    # Получаем предсказанные оценки
    predictions = model.predict([user_movie_pairs[:, 0], user_movie_pairs[:, 1],
                                 user_movie_pairs[:, 0], user_movie_pairs[:, 1]]).flatten()

    # Сортируем фильмы по предсказанным оценкам
    top_indices = predictions.argsort()[-n_recommendations:][::-1]
    top_movie_ids = unseen_movies[top_indices]
    top_ratings = predictions[top_indices]

    # Создаем DataFrame с рекомендациями
    recommendations = pd.DataFrame({
        'movie_idx': top_movie_ids,
        'predicted_rating': top_ratings
    }).merge(movies, on='movie_idx')

    return recommendations


# Функция для визуализации истории обучения
def plot_training_history(histories):
    plt.figure(figsize=(15, 6))

    # Loss
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['val_loss'], label=f'Fold {i + 1}', alpha=0.7)
    plt.title('Validation Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['val_mae'], label=f'Fold {i + 1}', alpha=0.7)
    plt.title('Validation MAE Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Функция для анализа рекомендаций
def analyze_recommendations(user_id, n_recommendations=10):
    # Получаем рекомендации
    recommendations = get_recommendations(user_id, n_recommendations)

    # Получаем топ фильмов, которые пользователь уже оценил
    user_ratings = ratings[ratings['user_idx'] == user_id]
    top_user_ratings = user_ratings.sort_values('rating_norm', ascending=False).head(n_recommendations)
    top_user_ratings = top_user_ratings.merge(movies, on='movie_idx')

    # Визуализация
    plt.figure(figsize=(14, 8))

    # Рекомендуемые фильмы
    plt.subplot(1, 2, 1)
    sns.barplot(x='predicted_rating', y='title', data=recommendations, palette='viridis')
    plt.title(f'Top {n_recommendations} Recommended Movies for User {user_id}')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Movie Title')

    # Любимые фильмы пользователя
    plt.subplot(1, 2, 2)
    sns.barplot(x='rating_norm', y='title', data=top_user_ratings, palette='magma')
    plt.title(f'Top {n_recommendations} Rated Movies by User {user_id}')
    plt.xlabel('Actual Rating')
    plt.ylabel('Movie Title')

    plt.tight_layout()
    plt.show()

    return recommendations, top_user_ratings


# Функция для оценки точности модели
def evaluate_model_accuracy(test_data):
    # Предсказания модели
    predictions = model.predict([test_data['user_idx'], test_data['movie_idx'],
                                 test_data['user_idx'], test_data['movie_idx']]).flatten()

    # Округление предсказаний для классификации
    predicted_ratings = np.round(predictions * 2) / 2  # Округляем до ближайшего 0.5

    # Матрица ошибок
    cm = confusion_matrix(test_data['rating_norm'] * 2, predicted_ratings * 2)

    # Визуализация матрицы ошибок
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.arange(0.5, 5.5, 0.5),
                yticklabels=np.arange(0.5, 5.5, 0.5))
    plt.title('Confusion Matrix for Rating Predictions')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Actual Rating')
    plt.show()

    # Расчет точности
    accuracy = np.mean(np.abs(predictions - test_data['rating_norm']) < 0.5)
    print(f"Model Accuracy (within ±0.5): {accuracy:.2%}")


# Пример использования
if __name__ == "__main__":
    # 1. Визуализация обучения
    plot_training_history(histories)

    # 2. Анализ рекомендаций для конкретного пользователя
    test_user_id = 42  # Пример пользователя
    recommendations, user_top_ratings = analyze_recommendations(test_user_id)

    print("\nTop Recommended Movies:")
    print(recommendations[['title', 'genres', 'predicted_rating']].to_string(index=False))

    print("\nUser's Top Rated Movies:")
    print(user_top_ratings[['title', 'genres', 'rating_norm']].to_string(index=False))

    # 3. Оценка точности на тестовых данных
    if len(test) > 0:
        evaluate_model_accuracy(test)
    else:
        print("No test data available for evaluation.")