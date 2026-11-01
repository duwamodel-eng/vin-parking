
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


BASE = Path(__file__).resolve().parent
MODEL = BASE / "runs/detect/train/weights/best.onnx"

CAMERA = "/dev/video0"       # ENTRY
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
TEST_SECONDS = 30

OUTPUT_IMAGE = BASE / "camera_entry_yolo.jpg"
OUTPUT_CROP = BASE / "camera_entry_plate_crop.jpg"


def letterbox(image, size=640):
    h, w = image.shape[:2]
    ratio = min(size / w, size / h)

    nw = int(round(w * ratio))
    nh = int(round(h * ratio))

    resized = cv2.resize(image, (nw, nh))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - nw) // 2
    top = (size - nh) // 2

    canvas[top:top + nh, left:left + nw] = resized

    return canvas, ratio, left, top


def detect(session, input_name, image):
    h, w = image.shape[:2]

    padded, ratio, pad_x, pad_y = letterbox(image, IMGSZ)

    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = blob.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    output = session.run(None, {input_name: blob})[0]

    # YOLOv8 one-class output: [1, 5, 8400]
    predictions = output[0].T

    boxes = []
    scores = []

    for row in predictions:
        cx, cy, bw, bh, confidence = row

        if not np.isfinite(row).all() or confidence < CONF_THRES:
            continue

        x1 = (cx - bw / 2 - pad_x) / ratio
        y1 = (cy - bh / 2 - pad_y) / ratio
        x2 = (cx + bw / 2 - pad_x) / ratio
        y2 = (cy + bh / 2 - pad_y) / ratio

        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(0, min(w - 1, int(round(x2))))
        y2 = max(0, min(h - 1, int(round(y2))))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(confidence))

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRES,
        IOU_THRES
    )

    if len(indices) == 0:
        return []

    results = []

    for index in np.array(indices).flatten():
        x, y, bw, bh = boxes[index]
        results.append({
            "box": (x, y, x + bw, y + bh),
            "confidence": scores[index]
        })

    return results


def main():
    if not MODEL.is_file():
        print("ERROR: Model not found:", MODEL)
        return

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(MODEL),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print("Model loaded:", MODEL.name)
    print("Provider:", session.get_providers())
    print("Opening ENTRY camera:", CAMERA)

    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    best_confidence = 0.0
    best_image = None
    best_box = None

    frame_count = 0
    detection_count = 0
    start = time.monotonic()
    last_report = start

    print(f"Testing for {TEST_SECONDS} seconds...")
    print("Keep a vehicle and its license plate visible to the camera.")

    try:
        while time.monotonic() - start < TEST_SECONDS:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("WARNING: Failed to read camera frame.")
                continue

            frame_count += 1

            # Lưu frame gốc trước khi vẽ bounding box
            raw_frame = frame.copy()

            detections = detect(session, input_name, frame)
            if detections:
                detection_count += 1

                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    conf = det["confidence"]

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                    cv2.putText(
                        frame,
                        f"Plate {conf:.3f}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    if conf > best_confidence:
                        best_confidence = conf
                        best_image = raw_frame.copy()
                        best_box = (x1, y1, x2, y2)

            now = time.monotonic()

            if now - last_report >= 2:
                elapsed = now - start
                print(
                    f"Time: {elapsed:.1f}s | "
                    f"Frames: {frame_count} | "
                    f"Detected frames: {detection_count} | "
                    f"Best confidence: {best_confidence:.3f}"
                )
                last_report = now

    finally:
        cap.release()

    elapsed = time.monotonic() - start

    print("\n========== TEST RESULT ==========")
    print("Elapsed:", round(elapsed, 2), "seconds")
    print("Frames processed:", frame_count)
    print("Frames with detection:", detection_count)
    print("Best confidence:", round(best_confidence, 4))

    if best_image is not None:
        cv2.imwrite(str(OUTPUT_IMAGE), best_image)

        x1, y1, x2, y2 = best_box
        crop = best_image[y1:y2, x1:x2]

        if crop.size > 0:
            cv2.imwrite(str(OUTPUT_CROP), crop)

        print("Annotated image:", OUTPUT_IMAGE)
        print("Plate crop:", OUTPUT_CROP)
        print("RESULT: DETECTION SUCCESS")
    else:
        print("RESULT: NO PLATE DETECTED")
        print("Check vehicle visibility, focus and camera angle.")


if __name__ == "__main__":
    main()
