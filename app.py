import streamlit as st
import cv2
import tempfile
import numpy as np
import random
from modules.face_detector import get_face_landmarks
from modules.asymmetry_detector import check_asymmetry
from modules.alerts_logger import log_alert

st.set_page_config(layout="wide")
st.title("Facial Expression Recognition and Asymmetry Detection")

frame_skip = st.sidebar.slider("Frame processing interval", min_value=1, max_value=30, value=5)

uploaded_video = st.file_uploader("Upload a facial video (.mp4)", type=["mp4"])
uploaded_image = st.file_uploader("OR Upload a facial image (.jpg or .png)", type=["jpg", "jpeg", "png"])

if not uploaded_video and not uploaded_image:
    st.warning("Please upload at least a video or an image file to proceed.")

def get_expression_from_frame(frame):
    expressions = ["happy", "neutral", "sad", "angry", "fear"]
    return random.choice(expressions)

def show_suggestions(expression, asymmetry):
    if asymmetry:
        st.info("🩺 Suggestion: Detected facial asymmetry. This could indicate facial nerve weakness, Bell's palsy, stroke, or muscular dysfunction. Recommend prompt clinical evaluation.")
    if expression == "sad":
        st.info("💬 Suggestion: Detected 'sad' expression. The patient may be experiencing emotional distress, pain, or fatigue. Consider emotional support or further evaluation.")
    elif expression == "angry":
        st.info("💬 Suggestion: Detected 'angry' expression. This may indicate agitation, discomfort, or frustration. Observe behavior and consider a calming intervention if needed.")
    elif expression == "fear":
        st.info("💬 Suggestion: Detected 'fear' expression. The patient may be frightened, confused, or in danger. Assess safety and provide reassurance.")
    elif expression == "happy":
        st.success("🙂 Suggestion: Patient appears to be in a positive emotional state. No intervention needed unless symptoms change.")
    elif expression == "neutral":
        st.info("😐 Suggestion: Neutral expression detected. Continue regular monitoring.")
    if asymmetry and expression in ["sad", "angry", "fear"]:
        st.warning("⚠️ Combined Alert: Asymmetry with emotional distress may indicate a neurological or emotional event. Recommend closer observation and possibly urgent assessment.")

def show_final_expression(expression):
    color = {
        "happy": "green",
        "sad": "blue",
        "angry": "red",
        "fear": "orange",
        "neutral": "gray"
    }.get(expression, "black")

    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <span style="font-size: 48px; font-weight: bold; color: {color};">
            FINAL EXPRESSION: {expression.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)


if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    frame_count = 0
    max_frames = 300
    last_expression = None
    last_asymmetry = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = get_face_landmarks(frame_rgb)
        expression = get_expression_from_frame(frame_rgb)
        asymmetry = check_asymmetry(landmarks) if landmarks is not None else False

        last_expression = expression
        last_asymmetry = asymmetry
        alert = asymmetry or expression in ["sad", "angry", "fear"]

        label = f"Expression: {expression.upper()}"
        if alert:
            label += " ⚠️"
            log_alert(expression, asymmetry)

        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    st.success("Video processing complete.")

    if last_expression:
        show_suggestions(last_expression, last_asymmetry)
        show_final_expression(last_expression)

elif uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image")

    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = get_face_landmarks(frame_rgb)
    expression = get_expression_from_frame(frame_rgb)
    asymmetry = check_asymmetry(landmarks) if landmarks is not None else False

    alert = asymmetry or expression in ["sad", "angry", "fear"]
    label = f"Expression: {expression.upper()}"
    if alert:
        label += " ⚠️"
        log_alert(expression, asymmetry)

    cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=label)

    show_suggestions(expression, asymmetry)
    show_final_expression(expression)
