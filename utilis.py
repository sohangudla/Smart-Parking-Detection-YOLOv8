import cv2
import numpy as np

def YOLO_Detection(model, frame, conf=0.3):
    """Run YOLOv8 detection and return boxes, classes, and names."""
    results = model(frame, conf=conf)
    boxes, classes = [], []
    names = model.names
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            boxes.append((x1, y1, x2, y2))
            classes.append(cls)
    return boxes, classes, names

def drawPolygons(frame, posList, detection_points=None):
    """Draw parking polygons (red if occupied, green if free)."""
    overlay = frame.copy()
    occupied = 0
    for pos in posList:
        pts = np.array(pos, np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0)  # Green by default
        if detection_points:
            for point in detection_points:
                if cv2.pointPolygonTest(pts, point, False) >= 0:
                    color = (0, 0, 255)  # Red if occupied
                    occupied += 1
                    break
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    return frame, occupied

def label_detection(frame, text, tbox_color, left, top, right, bottom):
    """Draw label and bounding box."""
    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), tbox_color, 2)
    label = str(text)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (int(left), int(top) - 20), (int(left) + tw + 5, int(top)), tbox_color, -1)
    cv2.putText(frame, label, (int(left) + 2, int(top) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
