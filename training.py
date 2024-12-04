# pip install ultralytics

import os
from ultralytics import YOLO

ROOT_DIR = 'путь к файлам train test valid'

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch

# Use the model
results = model.train(data=os.path.join(ROOT_DIR, "data.yaml"), epochs=150)  # train the model
