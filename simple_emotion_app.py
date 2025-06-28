import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import mediapipe as mp
from tensorflow.keras.models import load_model

# Set page configuration for a cleaner look
st.set_page_config(
    page_title="Simple Emotion Detector",
    page_icon="😊",
    layout="wide"
)

# Custom CSS for a cleaner interface and modern look
st.markdown("""
<style>
    .main {
        padding: 1rem 3rem;
        max-width: 900px;
        margin: 0 auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f9f9f9;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.2rem;
        background-color: #1E88E5;
        color: white;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1565c0;
        color: white;
    }
    h1, h2 {
        color: #1565c0;
        font-weight: 700;
    }
    .emotion-bar {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a photo.
2. Ensure your face is clearly visible and well-lit.
3. The app will detect your face and predict your emotion.
4. Results will be displayed with confidence scores.
""")

# App title with a clean, modern look
st.title("😊 Simple Emotion Detector")
st.markdown("Upload a photo to detect your emotion")

# Load the pre-trained model
@st.cache_resource
def load_pretrained_model():
    model_path = "pretrained_face_emotion_model.h5"
    model = load_model(model_path, compile=False)
    return model

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_emojis = ['😠', '🤢', '😨', '😄', '😢', '😲', '😐']

# Preprocess image for model input
def preprocess_face(face_img):
    gray_face = ImageOps.grayscale(face_img)
    resized_face = gray_face.resize((64, 64))
    normalized_face = np.array(resized_face).astype("float32") / 255.0
    expanded_face = np.expand_dims(normalized_face, axis=-1)
    return np.expand_dims(expanded_face, axis=0)

def main():
    model = load_pretrained_model()

    photo = st.file_uploader("📸 Upload your face photo", type=["jpg", "jpeg", "png"])

    if photo is not None:
        image = Image.open(photo).convert("RGB")
        img_width, img_height = image.size

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            img_np = np.array(image)
            img_bgr = img_np[:, :, ::-1]
            results = face_detection.process(img_bgr)

            if not results.detections:
                st.error("No face detected. Please try again with your face clearly visible.")
            else:
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()

                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * img_width)
                    y_min = int(bboxC.ymin * img_height)
                    box_width = int(bboxC.width * img_width)
                    box_height = int(bboxC.height * img_height)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_min + box_width)
                    y_max = min(img_height, y_min + box_height)

                    face_roi = image.crop((x_min, y_min, x_max, y_max))
                    processed_face = preprocess_face(face_roi)
                    prediction = model.predict(processed_face, verbose=0)[0]
                    emotion_index = np.argmax(prediction)
                    emotion = emotion_labels[emotion_index]
                    confidence = prediction[emotion_index] * 100
                    emoji = emotion_emojis[emotion_index]

                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="green", width=3)

                    label = f"{emoji} {emotion} ({confidence:.1f}%)"
                    text_size = font.getbbox(label)[2:4]
                    text_bg = (x_min, y_min - text_size[1] - 4, x_min + text_size[0] + 4, y_min)
                    draw.rectangle(text_bg, fill="green")
                    draw.text((x_min + 2, y_min - text_size[1] - 2), label, fill="white", font=font)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, use_column_width=True)

                with col2:
                    st.markdown(f"## {emoji} {emotion}")
                    st.progress(float(confidence/100))
                    st.markdown(f"**Confidence:** {confidence:.1f}%")

                    emotions_df = {
                        "Emotion": emotion_labels,
                        "Confidence": prediction * 100
                    }
                    st.bar_chart(emotions_df, x="Emotion", y="Confidence", use_container_width=True, height=250)

if __name__ == "__main__":
    main()
