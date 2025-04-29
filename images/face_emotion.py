import os
import random
import cv2
import pygame
import numpy as np
import time
from keras.models import model_from_json

# Load the emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion-to-Song Folder Mapping
emotion_to_folder = {
    'angry': "Party",
    'disgust': "Party",
    'fear': "Devotional",
    'happy': "Happy",
    'neutral': "Romantic",
    'sad': "Sad",
    'surprise': "Happy"
}

# Emotion Labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    """Preprocess image for emotion model."""
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def get_random_song(emotion):
    """Fetch a random song from the corresponding folder based on emotion."""
    folder = emotion_to_folder.get(emotion, "Happy")
    folder_path = os.path.join(os.getcwd(), folder)

    if os.path.exists(folder_path):
        songs = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        if songs:
            random.shuffle(songs)
            return os.path.join(folder_path, songs[0])
    return None

def play_song(emotion):
    """Play the selected song using pygame."""
    song_path = get_random_song(emotion)

    if song_path:
        print(f"🎶 Now Playing: {song_path}")
        pygame.mixer.init()
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait until the song finishes
            continue
    else:
        print("❌ No song found for this emotion!")

# Start Webcam for Emotion Detection
webcam = cv2.VideoCapture(0)

detected_emotion = None
while True:
    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            detected_emotion = labels[pred.argmax()]

            print(f"Detected Emotion: {detected_emotion}")
            cv2.putText(im, detected_emotion, (p - 10, q - 10), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            break  # Stop after detecting the first emotion
        
        cv2.imshow("Emotion Detection", im)
        cv2.waitKey(2000)  # Display result for 2 seconds before closing
        break  # Stop execution after detecting one emotion

webcam.release()
cv2.destroyAllWindows()

# Pause for 2 seconds before playing song
time.sleep(2)

if detected_emotion:
    play_song(detected_emotion)
