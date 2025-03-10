import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from keras.models import model_from_json
from keras_preprocessing.image import img_to_array
import plotly.express as px

# Load Emotion Detection Model
@st.cache_resource
def load_model():
    try:
        model_path = "emotion_model.json"
        weights_path = "emotion_model.h5"
        if not os.path.exists(model_path) or not os.path.exists(weights_path):
            raise FileNotFoundError("Model files not found.")
        with open(model_path, "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        return model
    except FileNotFoundError as e:
        st.error(str(e))
        return None

# Load Face Detector
@st.cache_resource
def load_face_detector():
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        st.error("Haar Cascade file not found.")
        return None
    return cv2.CascadeClassifier(cascade_path)

# Process Video and Collect Emotion Data
def process_video(video_path, output_path, skip_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    model = load_model()
    face_detector = load_face_detector()
    
    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear",
                      3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    frame_number = 0
    emotions_over_time = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % skip_frames != 0:
            frame_number += 1
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]
            
            emotions_over_time.append(label)
            timestamps.append(frame_number / fps)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    os.rename(temp_output_path, output_path)

    data = pd.DataFrame({"Timestamp": timestamps, "Emotion": emotions_over_time})
    return data

# Generate Visualizations
def generate_charts(data):
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.scatter(data, x="Timestamp", y="Emotion", title="Emotion Over Time"), use_container_width=True)
    with col2:
        emotion_counts = data["Emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        st.plotly_chart(px.pie(emotion_counts, names="Emotion", values="Count", title="Emotion Distribution"), use_container_width=True)

    st.plotly_chart(px.bar(emotion_counts, x="Emotion", y="Count", title="Emotion Frequency", labels={"Count": "Number of Frames"}), use_container_width=True)

# Streamlit UI
st.set_page_config(page_title="Emotion Detection Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Video", "Webcam Mode"])

if page == "Upload Video":
    st.title("🎥 Upload a Video for Emotion Detection")
    uploaded_file = st.file_uploader("Choose a Video File", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        
        st.info("Processing video... This may take some time ⏳")
        emotion_data = process_video(video_path, output_video_path, skip_frames=5)
        st.success("✅ Processing complete! See results below:")

        st.subheader("📽 Processed Video")
        st.video(output_video_path)

        st.subheader("📊 Emotion Analysis")
        generate_charts(emotion_data)

        st.subheader("📥 Download Processed Video")
        with open(output_video_path, "rb") as file:
            st.download_button("Download Video", file, file_name="processed_video.mp4")

elif page == "Webcam Mode":
    st.title("📸 Real-time Emotion Detection (Coming Soon)")
    st.warning("⚠️ Live webcam processing feature will be added in the next update!")

st.sidebar.markdown("---")
st.sidebar.write("💡 Built with ❤️ using Streamlit")
