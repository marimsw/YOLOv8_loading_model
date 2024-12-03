# pip install ultralytics

from ultralytics import YOLO

# Загрузка модели с весами
model = YOLO('/content/drive/MyDrive/Rabota/Razmetka_DATASET/best.pt')

# Выполнение детекции на изображении
results = model('/content/drive/MyDrive/Rabota/Razmetka_DATASET/660.jpg')

# Отображение результатов
for result in results:
    result.show()  # Отображение каждого результата
