import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load trained model
model = load_model("emotion_model.h5")

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    img = cv2.resize(img, (48,48))

    img = img / 255.0

    img = img.reshape(1,48,48,1)

    prediction = model.predict(img)

    emotion_index = np.argmax(prediction)

    emotion = emotion_labels[emotion_index]

    return jsonify({"Emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)