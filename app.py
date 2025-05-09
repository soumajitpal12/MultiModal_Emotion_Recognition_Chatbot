from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from chatbot import get_bot_response, detect_emotion_from_image
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from waitress import serve

# Load environment variables
load_dotenv()

app = Flask(__name__)
mail = Mail(app)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Upload folder
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Email config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = 'soumajit.bca@gmail.com'
mail = Mail(app)

# Load emotion model ONCE at startup
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotiondetector.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user'] = user.username
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error="Username already exists.")
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        msg = Message(f"New Contact Form Submission: {subject}", recipients=["soumajit.bca@gmail.com"])
        msg.body = f"""
        You have a new contact form submission:
        Name: {name}
        Email: {email}
        Subject: {subject}
        Message: {message}
        """
        try:
            mail.send(msg)
            print("Email sent successfully.")
        except Exception as e:
            print(f"Error sending email: {e}")
        return redirect(url_for('contact_thanks'))
    return render_template('home.html')

@app.route('/contact-thanks')
def contact_thanks():
    return render_template('contact_thanks.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = get_bot_response(user_input)
    return jsonify({"response": response})

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "input.jpg")
    file.save(filepath)
    emotion = detect_emotion_from_image(filepath)
    if emotion in ["No face found", "No image loaded"]:
        return jsonify({"response": "Sorry, I couldn't find a face in that image."})
    else:
        response = get_bot_response(f"I'm feeling {emotion}")
        return jsonify({"response": f"You look {emotion}. {response}"})

@app.route('/facial-emotion')
def facial_emotion():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('facial_emotion.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                prediction = emotion_model.predict(roi)
                label = emotion_labels[np.argmax(prediction)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    serve(app, host="0.0.0.0", port=8000, threads=8)
