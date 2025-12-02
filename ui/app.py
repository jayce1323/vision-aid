import streamlit as st
from PIL import Image
import uuid
import os

from object_detection.detector import ObjectDetector
from audio.tts_engine import tts_bytes


st.set_page_config(page_title="Vision-Aid", layout="centered")

st.title("🔍 Vision-Aid: Object Detection + Voice Assistance")
st.write("이미지 또는 카메라 입력을 이용해 객체를 인식하고 음성으로 안내합니다.")

detector = ObjectDetector()

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("카메라로 촬영하기")

image_path = None

if uploaded_file is not None:
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    image_path = temp_path
    st.image(uploaded_file, caption="업로드한 이미지", use_column_width=True)

elif camera_file is not None:
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(camera_file.getvalue())
    image_path = temp_path
    st.image(camera_file, caption="촬영된 이미지", use_column_width=True)

if image_path and st.button("🔎 분석하기"):
    with st.spinner("YOLO로 객체를 분석하는 중입니다..."):
        result = detector.detect_from_image(image_path)

    total = result["total_objects"]
    detections = result["detections"]

    if total > 0:
        labels = [d["label"] for d in detections]
        text_result = ", ".join(labels) + "가(이) 있습니다."
    else:
        text_result = "객체를 인식하지 못했습니다."

    st.success("분석 완료!")
    st.write("### 📌 인식 결과")
    st.write(text_result)

    with st.spinner("음성 변환 중입니다..."):
        audio_data = tts_bytes(text_result)

    st.write("### 🔊 음성 안내")
    st.audio(audio_data, format="audio/mp3")

    try:
        os.remove(image_path)
    except:
        pass
