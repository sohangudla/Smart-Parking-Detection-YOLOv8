import cv2
import torch
import pickle
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utilis import YOLO_Detection, drawPolygons, label_detection

# ========= Paths (keep all these files next to main.py) =========
BASE_DIR = Path(__file__).parent.resolve()

MODEL_PATH = BASE_DIR / "yolov8n.pt"          # put yolov8n.pt here
ROIS_PATH  = BASE_DIR / "Space_ROIs"          # put Space_ROIs here (the pickle saved by selector)
VIDEO_SRC  = BASE_DIR / "parking_space.mp4"   # or replace with your video filename in this folder

# ========= Sanity checks =========
def must_exist(p: Path, kind: str):
    if not p.exists():
        raise FileNotFoundError(f"{kind} not found: {p}\n"
                                f"→ Put the file in this folder or update the path.")
    return p

must_exist(MODEL_PATH, "Model")
must_exist(ROIS_PATH, "ROI pickle")
must_exist(VIDEO_SRC, "Video")

# ========= Device =========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Using: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ========= Load model =========
model = YOLO(str(MODEL_PATH))
model.to(device)

# ========= Load polygons =========
with open(ROIS_PATH, "rb") as f:
    posList = pickle.load(f)
if not isinstance(posList, list):
    # allow {'polys': [...]} structure too
    posList = posList.get('polys', [])

print(f"[ROIs] Loaded {len(posList)} polygons from: {ROIS_PATH.name}")

# ========= Open video =========
cap = cv2.VideoCapture(str(VIDEO_SRC))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_SRC}")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Video] {VIDEO_SRC.name} | {width}x{height}")

# ========= Main loop =========
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Video] End of stream.")
            break

        # YOLO detection
        boxes, classes, names = YOLO_Detection(model, frame)

        # centers for occupancy
        detection_points = []
        for (x1, y1, x2, y2) in boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detection_points.append((cx, cy))

        # color polygons & count occupancy
        frame, occupied_count = drawPolygons(frame, posList, detection_points=detection_points)

        total_slots   = len(posList)
        available_cnt = total_slots - occupied_count

        # HUD
        HUD_BG = (250, 250, 250)
        HUD_TX = (50, 50, 50)
        cv2.rectangle(frame, (int((width/2) - 200), 5), (int((width/2) - 40), 40), HUD_BG, -1)
        cv2.putText(frame, f"Fill Slots: {occupied_count}", (int((width/2) - 190), 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, HUD_TX, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (int(width/2), 5), (int((width/2) + 175), 40), HUD_BG, -1)
        cv2.putText(frame, f"Free Slots: {available_cnt}", (int((width/2) + 10), 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, HUD_TX, 1, cv2.LINE_AA)

        # draw boxes + labels
        for (x1, y1, x2, y2), cls in zip(boxes, classes):
            name = names[int(cls)] if isinstance(names, (list, tuple, dict)) else str(cls)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            inside = any(cv2.pointPolygonTest(np.array(p, np.int32), (cx, cy), False) >= 0 for p in posList)
            color  = (50, 50, 50) if inside else (100, 25, 50)
            label_detection(frame, text=str(name), tbox_color=color,
                            left=x1, top=y1, right=x2, bottom=y2)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), thickness=2)

        cv2.imshow("Parking Monitor (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
