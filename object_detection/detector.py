import cv2
import numpy as np

class ObjectDetector:
    def __init__(self,
                 weights="object_detection/yolov3.weights",
                 config="object_detection/yolov3.cfg",
                 names="object_detection/coco.names",
                 conf_threshold=0.5,
                 nms_threshold=0.4):

        self.net = cv2.dnn.readNet(weights, config)

        with open(names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_from_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    cx, cy, w, h = detection[0:4] * np.array([width, height, width, height])
                    x = int(cx - w/2)
                    y = int(cy - h/2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    "label": self.classes[class_ids[i]],
                    "confidence": confidences[i],
                    "bbox": {"x": x, "y": y, "w": w, "h": h}
                })

        return {
            "image_path": image_path,
            "detections": detections,
            "total_objects": len(detections)
        }


if __name__ == "__main__":
    detector = ObjectDetector()
    result = detector.detect_from_image("sample.jpg")
    print(result)
