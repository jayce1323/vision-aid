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

    def _process_detections(self, img, outs):
        height, width = img.shape[:2]
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
        return detections, boxes, confidences, class_ids

    def detect_from_array(self, img_array):
        if img_array is None:
            raise ValueError("Image array cannot be None")

        height, width = img_array.shape[:2]

        blob = cv2.dnn.blobFromImage(img_array, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.output_layers)

        detections, boxes, confidences, class_ids = self._process_detections(img_array, outs)

        return {
            "detections": detections,
            "total_objects": len(detections),
            "original_image": img_array
        }
    
    def draw_boxes(self, img, detections):
        if img is None:
            return None
        
        img_copy = img.copy()
        
        for d in detections:
            x, y, w, h = d["bbox"].values()
            label = d["label"]
            confidence = d["confidence"]
            
            color = (0, 255, 0)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_copy, f"{label} {confidence:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img_copy

    def detect_from_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.output_layers)
        
        detections, boxes, confidences, class_ids = self._process_detections(img, outs)

        return {
            "image_path": image_path,
            "detections": detections,
            "total_objects": len(detections)
        }
    
    def _process_detections(self, img, outs):
        height, width = img.shape[:2]
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
        return detections, boxes, confidences, class_ids

if __name__ == "__main__":
    detector = ObjectDetector()
    result = detector.detect_from_image("sample.jpg")
    print(result)