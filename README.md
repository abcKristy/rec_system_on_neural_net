# Movie Recommendation System with Neural Collaborative Filtering

Проект представляет собой систему рекомендаций фильмов на основе:
1. Нейросетевой коллаборативной фильтрации (NeuMF)
2. Классических методов (SVD, k-NN)
3. Оптимизированного подхода с кросс-валидацией

## Структура проекта
```bash
vkr/
├── data_preparation/
│ ├── prepare.py # Подготовка данных (нормализация, split)
│ ├── movie_to_idx.npy # Маппинг названий фильмов в индексы
│ ├── user_to_idx.npy # Маппинг пользователей в индексы
│ └── movies_enhanced.pk # Обогащенные метаданные фильмов
│
├── models/
│ ├── neuMF_final.h5 # Финальная версия модели
│ ├── neuMF_optimized.h5 # Оптимизированная версия
│ └── *.keras # Альтернативные форматы моделей
│
├── experiments/
│ ├── part1.py # Базовая реализация NeuMF
│ ├── part2.py # Модель с регуляризацией (борьба с переобучением)
│ ├── part3.py # Финальная версия с кросс-валидацией
│ └── part4.py # Сравнение с SVD/k-NN
│
├── evaluation/
│ ├── test.py # Основные тесты производительности
│ └── final_test.py # Финальное тестирование
│
└── results/
├── all_users_recommend # Готовые рекомендации
├── X_train_*.npy # Тренировочные данные (GMF/MLP)
└── y_*.npy # Целевые значения
```

## Ключевые особенности
1. `part1.py` - базовая реализация NeuMF:
   - Гибрид GMF + MLP
   - Минимальная регуляризация

2. `part2.py` - улучшенная версия:
   - Добавлен Dropout
   - L2-регуляризация
   - Early Stopping

3. `part3.py` - финальный вариант:
   - User-based кросс-валидация
   - Оптимизация гиперпараметров
   - Ensemble подход

4. `part4.py` - сравнение с baseline:
   - Точность/Recall@K
   - Время обучения
   - Качество на sparse данных

## Требования
- Python 3.8+
- TensorFlow 2.6+
- NumPy, Pandas, Scikit-learn

# Использование

## Подготовка данных
```bash
python prepare.py
```

## Запуск построения модели
```bash
python part3.py
```

## Тестирование
```bash
python test.py
```
