import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import mediapipe as mp
from pretrained_model_loader import load_pretrained_model

# Load pre-trained model
model = load_pretrained_model()

# Emotion labels for the pre-trained model (update if needed)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def preprocess_face(face_img):
    # Convert to grayscale
    gray_face = ImageOps.grayscale(face_img)
    # Resize to 64x64
    resized_face = gray_face.resize((64, 64))
    # Normalize pixel values
    normalized_face = np.array(resized_face).astype("float32") / 255.0
    # Expand dims for model input
    expanded_face = np.expand_dims(normalized_face, axis=-1)
    expanded_face = np.expand_dims(expanded_face, axis=0)
    return expanded_face

st.title("🎭 Facial Emotion Recognition - Photo Upload")

photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if photo is not None:
    image = Image.open(photo).convert("RGB")
    img_width, img_height = image.size

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Convert PIL image to numpy array
        img_np = np.array(image)
        # Convert RGB to BGR for mediapipe
        img_bgr = img_np[:, :, ::-1]
        results = face_detection.process(img_bgr)

        if not results.detections:
            st.write("No face detected in the photo.")
        else:
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * img_width)
                y_min = int(bboxC.ymin * img_height)
                box_width = int(bboxC.width * img_width)
                box_height = int(bboxC.height * img_height)

                # Ensure bounding box is within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_min + box_width)
                y_max = min(img_height, y_min + box_height)

                face_roi = image.crop((x_min, y_min, x_max, y_max))
                processed_face = preprocess_face(face_roi)
                prediction = model.predict(processed_face, verbose=0)
                emotion_index = np.argmax(prediction)
                emotion = emotion_labels[emotion_index] if emotion_index < len(emotion_labels) else "Unknown"
                label = f"Emotion: {emotion}"

                # Draw rectangle
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
                # Draw label above rectangle
                text_size = font.getbbox(label)[2:4]
                text_bg = (x_min, y_min - text_size[1] - 4, x_min + text_size[0] + 4, y_min)
                draw.rectangle(text_bg, fill="red")
                draw.text((x_min + 2, y_min - text_size[1] - 2), label, fill="white", font=font)

            st.image(image)
