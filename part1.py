# базовая модель с переобучением
import numpy as np
import pandas as pd
from keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# Нормализация оценок (исходный диапазон: 0.5 - 5)
ratings = pd.read_csv('ml-latest-small/ratings.csv')
min_rating = ratings['rating'].min()
max_rating = ratings['rating'].max()

# Загрузка данных
X_train_gmf = np.load('X_train_gmf.npy')
X_train_mlp = np.load('X_train_mlp.npy')
y_train = np.load('y_train.npy')
X_test_gmf = np.load('X_test_gmf.npy')
X_test_mlp = np.load('X_test_mlp.npy')
y_test = np.load('y_test.npy')

# Параметры датасета
num_users = len(np.unique(X_train_gmf[:, 0]))
num_movies = len(np.unique(X_train_gmf[:, 1]))
print(f"Число пользователей: {num_users}, Число фильмов: {num_movies}")

embedding_size = 32  # Размер эмбеддингов для пользователей и фильмов
mlp_layers = [64, 32, 16]  # Архитектура MLP-ветки

# Входы для GMF и MLP
input_gmf_user = Input(shape=(1,), name='gmf_user_input')
input_gmf_movie = Input(shape=(1,), name='gmf_movie_input')
input_mlp_user = Input(shape=(1,), name='mlp_user_input')
input_mlp_movie = Input(shape=(1,), name='mlp_movie_input')

# Эмбеддинги для GMF
gmf_user_embedding = Embedding(num_users, embedding_size, name='gmf_user_embedding')(input_gmf_user)
gmf_movie_embedding = Embedding(num_movies, embedding_size, name='gmf_movie_embedding')(input_gmf_movie)

# Элементное произведение (как в матричной факторизации)
gmf_user_flatten = Flatten()(gmf_user_embedding)
gmf_movie_flatten = Flatten()(gmf_movie_embedding)
gmf_output = Multiply()([gmf_user_flatten, gmf_movie_flatten])

# Эмбеддинги для MLP
mlp_user_embedding = Embedding(num_users, embedding_size, name='mlp_user_embedding')(input_mlp_user)
mlp_movie_embedding = Embedding(num_movies, embedding_size, name='mlp_movie_embedding')(input_mlp_movie)

# Конкатенация + полносвязные слои
mlp_user_flatten = Flatten()(mlp_user_embedding)
mlp_movie_flatten = Flatten()(mlp_movie_embedding)
mlp_concat = Concatenate()([mlp_user_flatten, mlp_movie_flatten])

for units in mlp_layers:
    mlp_concat = Dense(units, activation='relu')(mlp_concat)

# Конкатенация GMF и MLP
neuMF_concat = Concatenate()([gmf_output, mlp_concat])

# Финальный слой с сигмоидой (рейтинги нормализованы в [0, 1])
output = Dense(1, activation='sigmoid', name='output')(neuMF_concat)

model = Model(
    inputs=[input_gmf_user, input_gmf_movie, input_mlp_user, input_mlp_movie],
    outputs=output
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',  # MSE для регрессии
    metrics=['mae']
)

print(model.summary())


history = model.fit(
    [X_train_gmf[:, 0], X_train_gmf[:, 1], X_train_mlp[:, 0], X_train_mlp[:, 1]],
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(
        [X_test_gmf[:, 0], X_test_gmf[:, 1], X_test_mlp[:, 0], X_test_mlp[:, 1]],
        y_test
    )
)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Пример предсказания для первого тестового пользователя
sample_user = X_test_gmf[0, 0]
sample_movie = X_test_gmf[0, 1]
pred_rating = model.predict([np.array([sample_user]), np.array([sample_movie]),
                            np.array([sample_user]), np.array([sample_movie])])

# Денормализация рейтинга
pred_rating_denorm = pred_rating * (max_rating - min_rating) + min_rating
print(f"Предсказанный рейтинг: {pred_rating_denorm[0][0]:.2f}, Реальный рейтинг: {y_test[0] * (max_rating - min_rating) + min_rating:.2f}")