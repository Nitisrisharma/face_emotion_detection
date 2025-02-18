import numpy as np
import cv2
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Load the trained model
model_path = "models/emotion_model_svm.pkl"
if not os.path.exists(model_path):
    print("Trained model not found! Please run train_model.py first.")
    exit()

svm_model = joblib.load(model_path)

# Define emotion labels (ensure they match the training labels)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, (48, 48))  # Resize to 48x48
    img = img.reshape(1, -1) / 255.0  # Flatten and normalize
    return img

# Path to test image (Change this to the actual test image path)
test_image_path = "dataset/fer2013.csv/test/disgust/PrivateTest_807646.jpg"

if not os.path.exists(test_image_path):
    print(f"Test image not found: {test_image_path}")
    exit()

# Process and predict emotion
image = preprocess_image(test_image_path)
predicted_label = svm_model.predict(image)[0]

# Display the emotion
print(f"Predicted Emotion: {emotion_labels[predicted_label]}")
