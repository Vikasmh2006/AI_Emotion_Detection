AI Emotion Detection Project

Project Description:
This project uses Deep Learning and Computer Vision to detect human facial emotions in real time. 
The model is trained using the FER2013 dataset and predicts emotions such as angry, happy, sad, surprise, fear, disgust and neutral.

Technologies Used:
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

Dataset:
FER2013 Facial Emotion Dataset

--------------------------------------------------
- Set up project folder
- Created virtual environment
- Installed TensorFlow, OpenCV and required libraries
- Downloaded and organized FER2013 dataset
--------------------------------------------------
- Implemented image preprocessing
- Loaded dataset using ImageDataGenerator
- Created training and validation datasets
- Resized images to 48x48 grayscale
--------------------------------------------------
- Built Convolutional Neural Network (CNN) model
- Added Conv2D, MaxPooling, Dropout, and Dense layers
- Trained model on FER emotion dataset
- Achieved ~50% validation accuracy
- Saved trained model as emotion_model.h5
--------------------------------------------------
- Loaded trained CNN model
- Processed test images using OpenCV
- Predicted emotion using trained model
- Verified emotion detection output
--------------------------------------------------
- Implemented real-time emotion detection
- Used OpenCV for face detection
- Integrated trained CNN model with webcam input
- Displayed predicted emotions on live video feed
--------------------------------------------------
- Built Flask backend API
- Created /predict endpoint for emotion detection
- Integrated trained CNN model with Flask
- Tested API using Postman with image upload
- Successfully received emotion prediction response
--------------------------------------------------
