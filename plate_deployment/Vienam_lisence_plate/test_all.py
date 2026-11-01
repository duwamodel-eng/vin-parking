
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO
from vehicle_registry import find_vehicle

# ===== CAU HINH =====
BASE_DIR = Path(r"D:\Vienam_lisence_plate")
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
TEST_DIR = BASE_DIR / "4_cars"
OUTPUT_DIR = BASE_DIR / "results_4_cars"

CONF_THRESHOLD = 0.05
IMAGE_SIZE = 1280
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"

# ===== KIEM TRA DUONG DAN =====
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Khong tim thay thu muc anh: {TEST_DIR}")

# ===== KHOI TAO MODEL =====
print("Dang load YOLO...")
model = YOLO(str(MODEL_PATH))

print("Dang khoi tao EasyOCR tren GPU...")
reader = easyocr.Reader(["en"], gpu=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chi doc anh goc, bo qua anh crop
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
image_paths = sorted(
    p for p in TEST_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in extensions
    and "crop" not in p.stem.lower()
)

if not image_paths:
    raise SystemExit(f"Khong tim thay anh trong: {TEST_DIR}")

print(f"\nTim thay {len(image_paths)} anh goc.\n")

for image_path in image_paths:
    print("\n" + "=" * 65)
    print(f"ANH: {image_path.name}")

    image = cv2.imread(str(image_path))
    if image is None:
        print("Loi: khong doc duoc anh.")
        continue

    # ===== YOLO DETECT =====
    prediction = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device=0,
        verbose=False
    )[0]

    if prediction.boxes is None or len(prediction.boxes) == 0:
        print("YOLO: Khong tim thay bien so.")
        print("TRANG THAI: CHUA THE XAC MINH")
        continue

    # Chon box co confidence cao nhat
    best_index = int(prediction.boxes.conf.argmax().item())
    box = prediction.boxes[best_index]

    x1, y1, x2, y2 = (
        box.xyxy[0].cpu().numpy().astype(int).tolist()
    )
    yolo_conf = float(box.conf.item())

    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        print("Loi: crop bien so rong.")
        continue

    # ===== TIEN XU LY =====
    enlarged = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # ===== EASYOCR =====
    ocr_results = reader.readtext(
        enlarged,
        detail=1,
        allowlist=OCR_ALLOWLIST
    )

    # Sap xep cac vung ky tu tu trai sang phai
    ocr_results.sort(
        key=lambda item: min(point[0] for point in item[0])
    )

    plate_text = "".join(
        item[1].replace(" ", "").replace("\n", "")
        for item in ocr_results
    )

    ocr_conf = (
        sum(float(item[2]) for item in ocr_results)
        / len(ocr_results)
        if ocr_results else 0.0
    )

    # ===== TRA CUU XE / TAI XE =====
    vehicle = find_vehicle(plate_text)

    print(f"YOLO confidence : {yolo_conf:.3f}")
    print(f"OCR text        : {plate_text or 'KHONG DOC DUOC'}")
    print(f"OCR confidence  : {ocr_conf:.3f}")

    if vehicle:
        print(f"Ma xe           : {vehicle['car_id']}")
        print(f"Tai xe          : {vehicle['driver']}")
        print("Trang thai      : DA DANG KY")
    else:
        print("Ma xe           : ---")
        print("Tai xe          : ---")
        print("Trang thai      : KHONG KHOP BIEN SO")

    # ===== LUU CROP VA ANH KET QUA =====
    crop_path = OUTPUT_DIR / f"{image_path.stem}_crop.jpg"
    cv2.imwrite(str(crop_path), enlarged)

    annotated = image.copy()
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = plate_text if plate_text else "NO OCR"
    cv2.putText(
        annotated,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    result_path = OUTPUT_DIR / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(result_path), annotated)

    print(f"Anh crop        : {crop_path}")
    print(f"Anh ket qua     : {result_path}")

print("\n" + "=" * 65)
print("HOAN TAT KIEM TRA YOLO + EASYOCR + TRA CUU XE")
print(f"Ket qua nam tai: {OUTPUT_DIR}")
