from PIL import Image
import cv2
import numpy as np
import joblib
import time

# Load the trained model
model_path = "models/emotion_model_svm.pkl"
svm_model = joblib.load(model_path)

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.reshape(1, -1) / 255.0
    return face

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam successfully opened. Press 'q' to exit.")

frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    frame_counter += 1

    if frame_counter % 5 != 0:  # Process only every 5th frame to reduce workload
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        processed_face = preprocess_face(face)

        try:
            prediction = svm_model.predict(processed_face)[0]
            emotion = emotion_labels[prediction]
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            emotion = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert frame to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show the image using PIL
    pil_image = Image.fromarray(frame_rgb)
    pil_image.show()

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
