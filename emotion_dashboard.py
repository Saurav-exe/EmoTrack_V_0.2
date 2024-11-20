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
        emotion_model = model_from_json(loaded_model_json)
        emotion_model.load_weights(weights_path)
        return emotion_model
    except FileNotFoundError as e:
        st.error(str(e))
        return None

# Load Haar Cascade for Face Detection
@st.cache_resource
def load_face_detector():
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        st.error("Haar Cascade file not found.")
        return None
    return cv2.CascadeClassifier(cascade_path)

# Convert to H.264 Format for Browser Playback
def convert_to_h264(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

# Process Video and Collect Emotion Data
def process_video(video_path, output_path, skip_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video Writer for Output Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    emotion_model = load_model()
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

        # Skip frames to speed up processing
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

            preds = emotion_model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]
            
            # Save timestamp and emotion for visualization
            emotions_over_time.append(label)
            timestamps.append(frame_number / fps)

            # Draw Rectangle and Emotion Label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    # Convert the output video to H.264 format
    convert_to_h264(temp_output_path, output_path)
    os.remove(temp_output_path)  # Clean up temporary file

    # Create a DataFrame for visualizations
    data = pd.DataFrame({"Timestamp": timestamps, "Emotion": emotions_over_time})
    return data

# Generate Visualizations
def generate_time_series(data):
    fig = px.scatter(data, x="Timestamp", y="Emotion", title="Emotion Over Time",
                     labels={"Timestamp": "Time (s)", "Emotion": "Detected Emotion"})
    return fig

def generate_emotion_distribution(data):
    emotion_counts = data["Emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]
    fig = px.pie(emotion_counts, names="Emotion", values="Count", title="Emotion Distribution")
    return fig

def generate_emotion_bar(data):
    emotion_counts = data["Emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]
    fig = px.bar(emotion_counts, x="Emotion", y="Count", title="Emotion Frequency",
                 labels={"Count": "Number of Frames"})
    return fig

# Streamlit App
st.title("Emotion Detection and Analysis Dashboard")

uploaded_file = st.file_uploader("Upload a Video for Emotion Detection", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Set the output video path to the Downloads folder
    output_video_path = os.path.join("C:\\Users\\Admin\\Downloads", "analyzed_video.mp4")

    # Process the video and collect emotion data
    st.info("Processing the video... This might take some time.")
    emotion_data = process_video(video_path, output_video_path, skip_frames=5)

    st.success("Video processing complete. Displaying the output below:")
    
    # Display the processed video
    st.subheader("Analyzed Video")
    st.video(output_video_path)

    # Display emotion analysis visualizations
    st.subheader("Emotion Analysis Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(generate_time_series(emotion_data), use_container_width=True)
    with col2:
        st.plotly_chart(generate_emotion_distribution(emotion_data), use_container_width=True)

    st.plotly_chart(generate_emotion_bar(emotion_data), use_container_width=True)

    # Display Emotion Frequency Table
    st.subheader("Emotion Frequency Analysis")
    emotion_freq = emotion_data["Emotion"].value_counts().reset_index()
    emotion_freq.columns = ["Emotion", "Frequency"]
    st.table(emotion_freq)

    # Additional Statistics
    st.subheader("Additional Statistics")
    total_duration = len(emotion_data) * 5 / (30)  # Adjusted for skipped frames (assuming 30 FPS)
    st.write(f"Total Duration (processed): {total_duration:.2f} seconds")

    # Clean up the temporary input video
    os.remove(video_path)
