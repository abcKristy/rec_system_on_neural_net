# модель готовая к дальнейшей реализацией и 5ью фолдами


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Скрывает INFO сообщения

import pandas as pd
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

from part2 import build_model, test, ratings_filtered

# Подготовка данных для кросс-валидации
X = ratings_filtered[['user_idx', 'movie_idx']].values
y = ratings_filtered['rating_norm'].values

# Инициализация KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Списки для хранения результатов
fold_results = []
histories = []

# Кросс-валидация
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\n{'=' * 40}")
    print(f"Обучение fold {fold + 1}/5")
    print(f"{'=' * 40}")

    # Разделение данных
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Создание модели для текущего фолда
    model = build_model()

    # Коллбэки
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        min_delta=0.001
    )

    # Обучение
    history = model.fit(
        [X_train[:, 0], X_train[:, 1], X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=128,
        epochs=50,
        validation_data=(
            [X_val[:, 0], X_val[:, 1], X_val[:, 0], X_val[:, 1]],
            y_val
        ),
        callbacks=[early_stopping],
        verbose=1
    )
    histories.append(history)

    # Оценка на валидационном наборе
    val_loss, val_mae = model.evaluate(
        [X_val[:, 0], X_val[:, 1], X_val[:, 0], X_val[:, 1]],
        y_val,
        verbose=0
    )

    # Сохранение результатов
    fold_results.append({
        'fold': fold + 1,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'epochs': len(history.history['loss']),
        'best_val_loss': min(history.history['val_loss'])
    })

    print(f"\nFold {fold + 1} результаты:")
    print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
    print(f"Лучший Val Loss: {min(history.history['val_loss']):.4f}")
    print(f"Обучено эпох: {len(history.history['loss'])}")

# Анализ результатов кросс-валидации
print("\nИтоговые результаты кросс-валидации:")
results_df = pd.DataFrame(fold_results)
print(results_df)

print("\nСредние метрики по всем фолдам:")
print(f"Средний Val Loss: {results_df['val_loss'].mean():.4f} ± {results_df['val_loss'].std():.4f}")
print(f"Средний Val MAE: {results_df['val_mae'].mean():.4f} ± {results_df['val_mae'].std():.4f}")

# Визуализация обучения для всех фолдов
plt.figure(figsize=(15, 5))
for i, history in enumerate(histories):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], label=f'Fold {i + 1}')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_mae'], label=f'Fold {i + 1}')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')

plt.legend()
plt.tight_layout()
plt.show()

# Обучение финальной модели на всех данных
print("\nОбучение финальной модели на всех данных...")
final_model = build_model()

final_history = final_model.fit(
    [X[:, 0], X[:, 1], X[:, 0], X[:, 1]],
    y,
    batch_size=128,
    epochs=np.mean(results_df['epochs']).astype(int),  # Среднее количество эпох по фолдам
    verbose=1
)

# Сохранение финальной модели
final_model.save('neuMF_final.keras')
print("Финальная модель сохранена как 'neuMF_final.keras'")

# Оценка на тестовом наборе (если нужно)
if len(test) > 0:
    test_loss, test_mae = final_model.evaluate(
        [test['user_idx'], test['movie_idx'], test['user_idx'], test['movie_idx']],
        test['rating_norm'],
        verbose=0
    )
    print(f"\nРезультаты на тестовом наборе:")
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")