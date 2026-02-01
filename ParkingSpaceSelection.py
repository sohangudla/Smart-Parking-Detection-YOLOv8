import sys, os, pickle, cv2, numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python ParkingSpaceSelection.py <video_file_or_stem>")
    print("Example: python ParkingSpaceSelection.py carPark1.mp4")
    sys.exit(1)

arg = sys.argv[1]
stem = 'webcam' if arg.isdigit() else Path(arg).stem
ROIS_PATH = f'Space_ROIs_{stem}.pkl'
REF_IMAGE = f'ROI_Reference_{stem}.png'

if not os.path.exists(REF_IMAGE):
    print(f"❌ {REF_IMAGE} not found. Run main.py once for this video so it auto-creates the reference image.")
    sys.exit(1)

# Load existing (supports list/dict)
posList = []
base_w = base_h = None
if os.path.exists(ROIS_PATH):
    with open(ROIS_PATH, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'polys' in data:
        posList = data['polys']; base_w, base_h = data.get('base_w'), data.get('base_h')
    elif isinstance(data, list):
        posList = data

polygon_points = []

def save(img):
    h, w = img.shape[:2]
    with open(ROIS_PATH, 'wb') as f:
        pickle.dump({'polys': posList, 'base_w': w, 'base_h': h}, f)
    print(f"✅ Saved {len(posList)} slots for '{stem}' at base size {w}x{h} -> {ROIS_PATH}")

def mouse(event, x, y, flags, params):
    global polygon_points, posList
    img = params['img']
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        if len(polygon_points) == 4:
            posList.append(polygon_points.copy())
            polygon_points = []
            save(img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, poly in enumerate(posList):
            if cv2.pointPolygonTest(np.array(poly, np.int32), (x, y), False) >= 0:
                posList.pop(i); save(img); break

while True:
    img = cv2.imread(REF_IMAGE)
    if img is None:
        print(f"❌ Could not read {REF_IMAGE}")
        break

    # draw existing polygons
    for poly in posList:
        pts = np.array(poly, np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,0,255), 2)

    # draw in-progress points
    for pt in polygon_points:
        cv2.circle(img, pt, 5, (0,255,0), -1)

    title = f"Define Parking Slots for '{stem}' (Left-click corners, Right-click delete, Q=Save&Exit)"
    cv2.imshow(title, img)
    cv2.setMouseCallback(title, mouse, {'img': img})
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save(img)
        break

cv2.destroyAllWindows()
