import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Define paths to train and test directories
train_dir = "dataset/fer2013.csv/train"
test_dir = "dataset/fer2013.csv/test"

# Initialize lists for images and labels
X, y = [], []

# Helper function to load images from directories
def load_images_from_directory(directory, label):
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            # Load the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))  # Resize to 48x48 as required
            X.append(img)
            y.append(label)

# Load images from train directory
for label, emotion in enumerate(os.listdir(train_dir)):
    load_images_from_directory(os.path.join(train_dir, emotion), label)

# Convert to numpy arrays
X = np.array(X) / 255.0  # Normalize images
X = X.reshape(X.shape[0], -1)  # Flatten the images (48x48 -> 2304)
y = np.array(y)

# Encode labels to integers (SVM requires numeric labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM model
svm_model = SVC(kernel='linear', C=1)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(svm_model, 'models/emotion_model_svm.pkl')
print("Model training complete and saved!")
