from gtts import gTTS
from playsound import playsound
from pathlib import Path

LABEL_KO = {
    "person": "사람",
    "car": "자동차",
    "bus": "버스",
    "bicycle": "자전거",
    "dog": "강아지",
    "cat": "고양이",
}

def build_description(detections):
    if not detections:
        return "화면에서 인식된 물체가 없습니다."

    counts = {}
    for d in detections:
        label = d.get("label", "")
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1

    parts = []
    for label, cnt in counts.items():
        name_ko = LABEL_KO.get(label, label)
        parts.append(f"{name_ko} {cnt}개")

    body = ", ".join(parts)
    sentence = f"앞에 {body}가 있습니다."
    return sentence

def speak_korean(text, save_path="audio/output.mp3"):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tts = gTTS(text=text, lang="ko")
    tts.save(str(path))
    playsound(str(path))
