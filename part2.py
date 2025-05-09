# базовая модель с исправлением переобучения
import numpy as np
import pandas as pd
from keras import Model
from keras.src.callbacks import EarlyStopping
from keras.src.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Dense, Concatenate, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# Загрузка и подготовка данных
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Нормализация оценок
min_rating, max_rating = ratings['rating'].min(), ratings['rating'].max()
ratings['rating_norm'] = (ratings['rating'] - min_rating) / (max_rating - min_rating)

# Фильтрация (менее агрессивная)
user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()
ratings_filtered = ratings[
    ratings['userId'].isin(user_counts[user_counts >= 5].index) &
    ratings['movieId'].isin(movie_counts[movie_counts >= 5].index)
].copy()

# Переиндексация
user_to_idx = {uid: i for i, uid in enumerate(ratings_filtered['userId'].unique())}
movie_to_idx = {mid: i for i, mid in enumerate(ratings_filtered['movieId'].unique())}
ratings_filtered['user_idx'] = ratings_filtered['userId'].map(user_to_idx)
ratings_filtered['movie_idx'] = ratings_filtered['movieId'].map(movie_to_idx)

# Разделение данных
train, test = train_test_split(ratings_filtered, test_size=0.2, random_state=42)

# Оптимальные параметры модели
num_users = len(user_to_idx)
num_movies = len(movie_to_idx)
embedding_size = 32  # Оптимальный размер
mlp_layers = [64, 32]  # Более простая архитектура
dropout_rate = 0.3  # Увеличили dropout
l2_reg = 0.001  # Умеренная регуляризация


# Архитектура NeuMF с улучшенной регуляризацией
def build_model():
    # Входные слои
    input_gmf_user = Input(shape=(1,))
    input_gmf_movie = Input(shape=(1,))
    input_mlp_user = Input(shape=(1,))
    input_mlp_movie = Input(shape=(1,))

    # GMF ветка
    gmf_user_embed = Embedding(num_users, embedding_size,
                               embeddings_regularizer=l2(l2_reg))(input_gmf_user)
    gmf_movie_embed = Embedding(num_movies, embedding_size,
                                embeddings_regularizer=l2(l2_reg))(input_gmf_movie)
    gmf_multiply = Multiply()([Flatten()(gmf_user_embed), Flatten()(gmf_movie_embed)])

    # MLP ветка с BatchNorm и Dropout
    mlp_user_embed = Embedding(num_users, embedding_size,
                               embeddings_regularizer=l2(l2_reg))(input_mlp_user)
    mlp_movie_embed = Embedding(num_movies, embedding_size,
                                embeddings_regularizer=l2(l2_reg))(input_mlp_movie)
    mlp_concat = Concatenate()([Flatten()(mlp_user_embed), Flatten()(mlp_movie_embed)])

    for units in mlp_layers:
        mlp_concat = Dense(units, activation='relu',
                           kernel_regularizer=l2(l2_reg))(mlp_concat)
        mlp_concat = Dropout(dropout_rate)(mlp_concat)

    # Объединение с дополнительным Dropout
    concat = Concatenate()([gmf_multiply, mlp_concat])
    concat = Dropout(dropout_rate)(concat)
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(
        inputs=[input_gmf_user, input_gmf_movie, input_mlp_user, input_mlp_movie],
        outputs=output
    )

    model.compile(
        optimizer=Adam(0.001),  # Стандартный learning rate
        loss=MeanSquaredError(),
        metrics=['mae']
    )
    return model


model = build_model()
model.summary()

# Обучение с тщательным мониторингом
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

history = model.fit(
    [train['user_idx'], train['movie_idx'], train['user_idx'], train['movie_idx']],
    train['rating_norm'],
    batch_size=128,
    epochs=50,
    validation_data=(
        [test['user_idx'], test['movie_idx'], test['user_idx'], test['movie_idx']],
        test['rating_norm']
    ),
    callbacks=[early_stopping],
    verbose=1
)

# Визуализация
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Evolution')
plt.legend()
plt.tight_layout()
plt.show()

# Проверка разнообразия предсказаний
sample_preds = model.predict([
    np.random.choice(num_users, 10),
    np.random.choice(num_movies, 10),
    np.random.choice(num_users, 10),
    np.random.choice(num_movies, 10)
])
print("Разброс предсказаний:", np.unique(sample_preds.round(2)))


# Пример предсказания для первого тестового пользователя (новая модель)
sample_user_idx = test['user_idx'].iloc[0]
sample_movie_idx = test['movie_idx'].iloc[0]
sample_rating_norm = test['rating_norm'].iloc[0]

# Получаем предсказание
pred_rating_norm = model.predict([
    np.array([sample_user_idx]),
    np.array([sample_movie_idx]),
    np.array([sample_user_idx]),
    np.array([sample_movie_idx])
], verbose=0)

# Денормализация рейтинга
pred_rating = pred_rating_norm[0][0] * (max_rating - min_rating) + min_rating
true_rating = sample_rating_norm * (max_rating - min_rating) + min_rating

print(f"\nТестовый пример:")
print(f"Пользователь ID: {test['userId'].iloc[0]}")
print(f"Фильм: {movies[movies['movieId'] == test['movieId'].iloc[0]]['title'].values[0]}")
print(f"Предсказанный рейтинг: {pred_rating:.2f}")
print(f"Реальный рейтинг: {true_rating:.2f}")
print(f"Ошибка предсказания: {abs(pred_rating - true_rating):.2f}")


# Сохранение модели
model.save('neuMF_optimized.keras')
np.save('user_to_idx.npy', user_to_idx)
np.save('movie_to_idx.npy', movie_to_idx)

