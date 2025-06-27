import streamlit as st
import numpy as np
import cv2
from pretrained_model_loader import load_pretrained_model

# Load pre-trained model
model = load_pretrained_model()

# Emotion labels for the pre-trained model (update if needed)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_frame(frame):
    gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (48, 48))
    normalized_face = resized_face.astype("float32") / 255.0
    expanded_face = np.expand_dims(normalized_face, axis=-1)
    return expanded_face

st.title("🎭 Facial Emotion Recognition - Photo Capture")

photo = st.camera_input("Capture a photo")

if photo is not None:
    img = np.array(bytearray(photo.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        st.write("No face detected in the photo.")
    else:
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            processed = preprocess_frame(face_roi)
            input_img = np.expand_dims(processed, axis=0)
            prediction = model.predict(input_img, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index] if emotion_index < len(emotion_labels) else "Unknown"
            label = f"Emotion: {emotion}"

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.image(img, channels="BGR")
