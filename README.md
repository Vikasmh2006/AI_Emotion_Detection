# AI Emotion Detection System 

This project detects human emotions from facial images using a Deep Learning Convolutional Neural Network (CNN).  
Users can upload an image and the system predicts the emotion.

---

## Features

- Detects 7 human emotions
- Uses CNN deep learning model
- Image upload from web interface
- Real-time emotion detection using OpenCV
- Flask API backend
- Colorful HTML/CSS frontend

---

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- HTML
- CSS
- JavaScript

---

## Project Structure

```
AI_Emotion_Detection
│
├── train_model.py
├── real_time_detection.py
├── app.py
├── emotion_model.h5
│
├── frontend
│   ├── index.html
│   └── style.css
│
└── dataset
```

---

## How It Works

```
User Upload Image
        ↓
Frontend (HTML + JS)
        ↓
Flask API (/predict)
        ↓
CNN Deep Learning Model
        ↓
Emotion Prediction
        ↓
Displayed on Webpage
```

---

## Dataset

FER2013 Facial Expression Dataset

---

## Future Improvements

- Deploy project online
- Improve CNN accuracy
- Add real-time webcam detection in web app
- Use larger datasets

---

## Screenshots

### Homepage
![Homepage](screenshots/homepage.png)

### Emotion Detection Result
![Result](screenshots/result.png)



## Author

Vikas MH  
AI & Data Science Student
