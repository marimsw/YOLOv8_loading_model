from ultralytics import YOLO
import multiprocessing
import torch

# Очистка кэша CUDA
torch.cuda.empty_cache()

# Инициализация модели YOLOv8
model = YOLO('yolov8m')

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Обучение модели с параметрами
    results = model.train(
        data='data.yaml',  # Путь к файлу с данными
        epochs=150,          # Общее количество эпох
        batch=16,            # Размер батча
        patience=30,         # Количество эпох без улучшения для ранней остановки
        name='yolov8m_for_basketboll'  # Имя для сохранения результатов
    )
