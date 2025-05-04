from groq import Groq
import os
import cv2
import numpy as np
from keras.models import model_from_json
from gtts import gTTS
import time

# Load chatbot API
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Chat context
chat_history = [{
    "role": "system",
    "content": (
        "You're a warm, friendly emotional support assistant. Respond like a supportive friend. "
        "Keep responses under 3 lines. Use a conversational tone like in a WhatsApp chat. "
        "Make the user feel heard, give emotional comfort, and gently guide them."
    )
}]

def get_bot_response(user_input):
    chat_history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        messages=chat_history,
        model="llama3-70b-8192"
    )
    reply = response.choices[0].message.content.strip()
    chat_history.append({"role": "assistant", "content": reply})
    return reply

# Load Emotion Detection Model
with open("emotiondetect.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetect.h5")

# Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

if face_cascade.empty():
    print("ERROR: Haar Cascade failed to load!")


labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion_from_image(image_path):
    print(f"DEBUG: Trying to load image from {image_path}")
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if gray is None:
        print("DEBUG: Failed to load image.")
        return "No image loaded"

    print("DEBUG: Image loaded successfully, detecting faces...")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    print(f"DEBUG: Number of faces detected = {len(faces)}")

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            try:
                face_img = cv2.resize(face_img, (48, 48))
            except Exception as e:
                print(f"DEBUG: Resize failed: {e}")
                continue
            processed = extract_features(face_img)
            prediction = model.predict(processed)
            emotion = labels[prediction.argmax()]
            print(f"DEBUG: Emotion detected = {emotion}")
            return emotion
    else:
        print("DEBUG: No faces found in image.")
    return "No face found"


# ✅ Generate speech from text
def generate_speech(text, folder):
    tts = gTTS(text)
    filename = f"response_{int(time.time())}.mp3"
    filepath = os.path.join(folder, filename)
    tts.save(filepath)
    return filename
