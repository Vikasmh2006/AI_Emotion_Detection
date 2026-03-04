import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load test image
img = cv2.imread("test/happy/PrivateTest_10077120.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to 48x48
gray = cv2.resize(gray, (48,48))

# Normalize
gray = gray / 255.0

# Reshape for CNN
gray = gray.reshape(1,48,48,1)

# Predict emotion
prediction = model.predict(gray)

emotion_index = np.argmax(prediction)

print("Predicted Emotion:", emotion_labels[emotion_index])