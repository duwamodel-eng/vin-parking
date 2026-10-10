import os
import re
import time
import cv2
import numpy as np
import onnxruntime as ort
import pytesseract

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INPUT_SIZE = 640
CONFIDENCE = 0.38
NMS_THRESHOLD = 0.45
OCR_EVERY_N_FRAMES = 3
ALLOWED_CHARS = "0123456789ABCDEFGHKLMNPSTUVXYZ"

if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"[LOI] Khong tim thay model: {MODEL_PATH}")

print("[INFO] Dang tai ONNX model...")
session = ort.InferenceSession(
    MODEL_PATH, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
print("[INFO] Input:", session.get_inputs()[0].shape)
print("[INFO] Output:", session.get_outputs()[0].shape)
print("[INFO] OCR:", pytesseract.get_tesseract_version())

def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def detect_plates(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = round(w * scale), round(h * scale)

    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    px, py = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas[py:py+nh, px:px+nw] = resized

    image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    output = session.run(None, {input_name: tensor})[0]
    pred = output[0]

    # YOLO detect output [4 + so lop, so du doan].
    # Model nay co 1 lop license-plate.
    if pred.shape[0] == 5:
        pred = pred.T
    elif pred.shape[1] != 5:
        raise RuntimeError(f"Output khong dung dang: {output.shape}")

    boxes, scores = [], []

    for row in pred:
        cx, cy, bw, bh, score = map(float, row)
        if not np.isfinite(row).all() or score < CONFIDENCE:
            continue
        if bw <= 0 or bh <= 0:
            continue

        x1 = int((cx - bw / 2 - px) / scale)
        y1 = int((cy - bh / 2 - py) / scale)
        x2 = int((cx + bw / 2 - px) / scale)
        y2 = int((cy + bh / 2 - py) / scale)

        x1, x2 = max(0, min(w-1, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h-1, y1)), max(0, min(h, y2))

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(score)

    if not boxes:
        return []

    ids = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE, NMS_THRESHOLD)
    return [(boxes[int(i)], scores[int(i)])
            for i in np.array(ids).reshape(-1)] if len(ids) else []

def read_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3,
                      interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    config = (
        "--oem 3 --psm 7 "
        f"-c tessedit_char_whitelist={ALLOWED_CHARS}"
    )
    return clean_text(pytesseract.image_to_string(gray, config=config))

print(f"[INFO] Dang mo webcam /dev/video{CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    cap.release()
    raise SystemExit("[LOI] Khong mo duoc webcam.")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

frame_count = 0
last_plate = ""
last_print = 0
prev_time = time.monotonic()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[CANH BAO] Khong doc duoc frame.")
            time.sleep(0.2)
            continue

        frame_count += 1
        now = time.monotonic()
        elapsed = now - prev_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        prev_time = now

        display = frame.copy()
        detections = detect_plates(frame)

        for (x, y, w, h), score in detections:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, f"Plate {score:.3f}",
                        (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)

        if detections and frame_count % OCR_EVERY_N_FRAMES == 0:
            for (x, y, w, h), score in detections:
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                text = read_plate(crop)
                if 6 <= len(text) <= 10:
                    now = time.monotonic()
                    if text != last_plate or now-last_print >= 3:
                        print(f"[BIEN SO] {text} | confidence={score:.3f}")
                        last_plate, last_print = text, now

        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2)
        cv2.putText(display, f"Detections: {len(detections)}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2)

        # cv2.imshow("Vin Parking - Camera", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

except KeyboardInterrupt:
    print("\n[INFO] Da dung chuong trinh.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Da dong webcam.")
