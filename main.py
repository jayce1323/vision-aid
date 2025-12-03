from object_detection.detector import detect_objects
from audio.tts_engine import build_description, speak_korean

def main():
    image_path = "images/street.jpeg"
    detections = detect_objects(image_path)
    description = build_description(detections)
    print("[시각장애인 안내 메시지]")
    print(description)
    speak_korean(description)

if __name__ == "__main__":
    main()
