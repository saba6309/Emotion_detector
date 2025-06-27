import streamlit as st
import numpy as np
import cv2
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
    .stCameraInput > div > div > div > button {
        border-radius: 8px;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        height: 3rem;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .stCameraInput > div > div > div > button:hover {
        background-color: #1565c0;
        color: white;
    }
    .emotion-bar {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Click the **Capture** button to take a photo.
2. Ensure your face is clearly visible and well-lit.
3. The app will detect your face and predict your emotion.
4. Results will be displayed with confidence scores.
""")

# App title with a clean, modern look
st.title("😊 Simple Emotion Detector")
st.markdown("Take a photo to detect your emotion")

# Load the pre-trained model
@st.cache_resource
def load_pretrained_model():
    model_path = "pretrained_face_emotion_model.h5"
    model = load_model(model_path, compile=False)
    return model

# Load face detection cascade
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_emojis = ['😠', '🤢', '😨', '😄', '😢', '😲', '😐']

# Preprocess image for model input
def preprocess_face(face_img):
    # Convert to grayscale
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Resize to model input size
    resized_face = cv2.resize(gray_face, (64, 64))
    # Normalize pixel values
    normalized_face = resized_face.astype("float32") / 255.0
    # Add channel dimension
    expanded_face = np.expand_dims(normalized_face, axis=-1)
    # Add batch dimension
    return np.expand_dims(expanded_face, axis=0)

# Main app logic
def main():
    model = load_pretrained_model()
    face_cascade = load_face_cascade()
    
    # Camera input with a clear instruction
    photo = st.camera_input("📸 Capture your face")
    
    if photo is not None:
        # Convert image to OpenCV format
        img_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Process results
        if len(faces) == 0:
            st.error("No face detected. Please try again with your face clearly visible.")
        else:
            # Create columns for results
            col1, col2 = st.columns(2)
            
            # Process each detected face
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = img[y:y+h, x:x+w]
                
                # Preprocess for model
                processed_face = preprocess_face(face_roi)
                
                # Predict emotion
                prediction = model.predict(processed_face, verbose=0)[0]
                emotion_index = np.argmax(prediction)
                emotion = emotion_labels[emotion_index]
                confidence = prediction[emotion_index] * 100
                emoji = emotion_emojis[emotion_index]
                
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display results
            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Face", use_container_width=True)

            with col2:
                st.markdown(f"## {emoji} {emotion}")
                st.progress(float(confidence/100))
                st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                # Show all emotions as a bar chart with custom styling
                emotions_df = {
                    "Emotion": emotion_labels,
                    "Confidence": prediction * 100
                }
                st.bar_chart(emotions_df, x="Emotion", y="Confidence", use_container_width=True, height=250)

if __name__ == "__main__":
    main()
