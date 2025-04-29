from tensorflow.keras.models import model_from_json
import cv2
import numpy as np

# Load model architecture
with open("emotiondetector.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("emotiondetector.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
