# Vision-Aid: 시각장애인을 위한 시각 보조 AI

## 1. 프로젝트 개요 (Project Overview)
**Vision-Aid**는 시각장애인 및 저시력자가 주변 환경을 독립적으로 인지할 수 있도록 돕는 오픈소스 소프트웨어입니다.
노트북이나 웹캠을 통해 실시간으로 사물을 인식(Object Detection)하고, 인식된 객체 정보를 **한국어 음성(TTS)**으로 변환하여 안내합니다. **YOLOv3** 알고리즘과 **Streamlit** 웹 프레임워크를 사용하여 개발되었으며, 직관적인 UI를 통해 누구나 쉽게 사용할 수 있습니다.

## 2. 데모 및 실행 예시 (Demo)
사용자가 이미지를 업로드하거나 카메라로 촬영하면, AI가 사물을 분석하여 바운딩 박스로 표시하고 음성으로 결과를 알려줍니다.

https://github.com/user-attachments/assets/252290d2-5b59-40ac-b2e2-3aa67869403b

## 3. 설치 및 사용 환경 (Prerequisites)
이 프로젝트는 **Python 3.8 이상** 환경에서 동작합니다.

### 사용된 패키지 및 버전
* `streamlit`: 사용자 인터페이스(UI) 구성
* `opencv-python`: 이미지 처리 및 YOLO 객체 탐지
* `numpy`: 데이터 연산 처리
* `Pillow`: 이미지 파일 처리
* `gtts`: 텍스트 음성 변환 (Text-to-Speech)
* `playsound`: 오디오 파일 재생 지원

## 4. 설치 및 실행 방법 (How to Run)

### Step 1. 프로젝트 저장소 복제 (Clone)
먼저 GitHub 저장소를 로컬 컴퓨터로 다운로드하고 해당 폴더로 이동합니다.
```bash
git clone https://github.com/jayce1323/vision-aid.git
cd vision-aid
```

### Step 2. 필수 패키지 설치 (Install Requirements)
프로젝트 실행에 필요한 라이브러리들을 일괄 설치합니다. 터미널에 아래 명령어를 입력하세요.
```bash
pip install streamlit opencv-python numpy Pillow gtts playsound
```

### Step 3. 모델 가중치 파일 다운로드 (중요)
주의: YOLOv3 모델의 핵심인 yolov3.weights 파일은 100MB를 초과하여 GitHub 저장소에 포함되어 있지 않습니다. 반드시 아래 공식 링크에서 파일을 다운로드한 후, object_detection 폴더 안에 넣어주셔야 프로그램이 정상 작동합니다.

파일명: yolov3.weights

다운로드 링크: https://pjreddie.com/media/files/yolov3.weights

저장 위치: vision-aid/object_detection/yolov3.weights

### Step 4. 프로그램 실행 (Run)
모든 설치와 파일 배치가 완료되었다면, 아래 명령어로 프로그램을 실행합니다.
```bash
streamlit run main.py
```
명령어를 실행하면 자동으로 웹 브라우저가 열리며(http://localhost:8501), Vision-Aid 서비스를 바로 사용할 수 있습니다.

## Team Members

| 역할 구분 | 이름 | 구체적인 기여 및 담당 모듈 |
| :--- | :--- | :--- |
| **팀 리더 / DevOps** | 안준영 | 프로젝트 총괄, **Gitflow 관리** (Merge, Conflict Resolution), Streamlit UI 통합, **YOLO 모델 로딩 최적화** (`@st.cache_resource` 적용), README 문서 작성. |
| **핵심 모델 구현** | 장윤겸 | **객체 탐지(YOLOv3) 모듈 구현** (`detector.py`), OpenCV DNN 로직 구축, NMS(Non-Maximum Suppression) 및 바운딩 박스 좌표 계산 로직 담당. |
| **음성 및 전처리** | 김동연 | **TTS(음성 합성) 모듈 구현** (`tts_engine.py`), `gtts`를 활용한 한국어 음성 생성, Streamlit 오디오 출력 포맷 (`BytesIO`) 처리 담당. |
| **데이터 및 UI 보조** | 박혜주 | **입력 이미지 전처리 및 후처리 로직** 담당, `coco.names` 데이터 관리, Streamlit **카메라 입력 및 파일 업로드** UI 보조 구현. |
