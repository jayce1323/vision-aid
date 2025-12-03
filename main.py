import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from object_detection.detector import ObjectDetector 
from audio.tts_engine import build_description 

def tts_bytes(text):
    from gtts import gTTS
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang="ko")
    tts.write_to_fp(mp3_fp)
    return mp3_fp.getvalue()

@st.cache_resource
def load_detector():
    """ObjectDetector 객체를 생성하고 캐싱합니다."""
    return ObjectDetector()

def main():
    st.set_page_config(page_title="Vision-Aid", layout="centered")
    st.title("시각장애인을 위한 시각 보조 AI")
    st.markdown("이미지를 업로드하면 **무엇이 있는지 음성으로 안내합니다.**")
    
    detector = load_detector()

    uploaded_file = st.file_uploader("이미지 파일 업로드", type=['jpg', 'png', 'jpeg'])
    camera_file = st.camera_input("카메라로 촬영하기")

    image_data = uploaded_file or camera_file

    if image_data is not None:
        
        image = Image.open(image_data)
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if st.button("분석 시작"):
            
            st.subheader("분석 결과")
            with st.spinner('AI가 객체를 분석하고 있습니다...'):

                result = detector.detect_from_array(image_cv) 
                detections = result["detections"]

                result_img_cv = detector.draw_boxes(image_cv, detections)
                
                result_img_rgb = cv2.cvtColor(result_img_cv, cv2.COLOR_BGR2RGB)

                text_result = build_description(detections)
                audio_data = tts_bytes(text_result)

            st.image(result_img_rgb, caption="감지된 사물", use_column_width=True)
            st.success(f"인식 텍스트: {text_result}")
            
            st.write("### 음성 안내")
            st.audio(audio_data, format="audio/mp3")
            
    elif image_data is not None:
         st.image(image, caption="분석 대기 이미지", use_column_width=True)


if __name__ == "__main__":
    main()