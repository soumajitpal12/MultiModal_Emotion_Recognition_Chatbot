# Multi-Modal Emotion Recognition Chatbot

This project integrates **Facial**, **Text**, and **Voice** inputs to detect human emotions and responds accordingly using a chatbot interface. It supports live webcam emotion detection, speech-to-text processing, sentiment analysis, and chatbot interaction with voice output.

## Features

- Real-time facial emotion recognition using webcam or image upload  
- Voice emotion detection through tone and text analysis  
- Text sentiment analysis and emotion detection  
- Integrated chatbot that replies based on detected mood  
- Supports voice input and voice output  
- WhatsApp-like frontend with chat history  

## Project Structure

```
MultiModal_Emotion_Recognition_Chatbot/
├── app.py                      # Flask backend
├── chatbot.py                  # Chat logic, emotion response, TTS
├── emotiondetector.json        # Model structure for facial emotion
├── emotiondetector.h5          # Model weights for facial emotion
├── static/
│   └── style.css               # Frontend styling
├── templates/
│   └── index.html              # Chat interface
├── database/
│   └── chat.db                 # (Optional) SQLite chat history
├── requirements.txt            # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/soumajitpal12/MultiModal_Emotion_Recognition_Chatbot.git
cd MultiModal_Emotion_Recognition_Chatbot
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

The app will start at: [http://localhost:5000](http://localhost:5000)

## Usage

1. Open your browser and navigate to `http://localhost:5000`  
2. Use the chat interface to:
   - Type messages  
   - Upload voice input or images  
   - Click on **Detect Emotion** from webcam  
3. The chatbot will respond based on the detected emotion with voice and text output.

## Model Details

- **Facial Emotion Model**: CNN trained on FER-2013 dataset  
- **Voice/Text Emotion**: Based on tone analysis and sentiment polarity  

## Future Improvements

- Deploy with PlanetScale or cloud database  
- Add support for multilingual emotion detection  
- Improve emotion classification accuracy using transformers  

## Author

Soumajit Pal

## License

This project is licensed under the MIT License.