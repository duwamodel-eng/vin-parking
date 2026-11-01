from pathlib import Path
from collections import defaultdict
from itertools import combinations
import re
import time

import cv2
import easyocr
from ultralytics import YOLO
from vehicle_registry import find_vehicle, VEHICLES


# =========================================================
# 1. CAU HINH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
OUTPUT_DIR = BASE_DIR / "camera_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 1
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

IMAGE_SIZE = 960
CONF_THRESHOLD = 0.20

OCR_USE_GPU = False
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

PADDING_RATIO_X = 0.05
PADDING_RATIO_Y = 0.10

# Chi phong to khi crop nho; crop da lon thi giu nguyen
TARGET_CROP_WIDTH = 400
MAX_UPSCALE = 3.0

# Burst: lay N frame, giu K frame net nhat de OCR
BURST_FRAMES = 6
BURST_KEEP = 3
BURST_DELAY = 0.08          # giay giua 2 frame

# Lay danh sach bien so truc tiep tu vehicle_registry.py
REGISTERED_PLATES = list(VEHICLES.keys())

MAX_REGISTRY_DIST = 1.0     # xa hon nguong nay thi khong tinh la khop
MIN_AMBIGUITY_GAP = 0.5     # 2 bien gan nhat phai cach nhau it nhat bay nhieu
MIN_AGREEMENT = 0.8         # ti le phieu toi thieu de coi la "dong thuan"


# =========================================================
# 2. TAI MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay model YOLO: {MODEL_PATH}")

print("Dang tai YOLO...")
model = YOLO(str(MODEL_PATH))

print("Dang tai EasyOCR...")
reader = easyocr.Reader(["en"], gpu=OCR_USE_GPU)

print("Tai model hoan tat.")


# =========================================================
# 3. MO CAMERA - WINDOWS DIRECTSHOW
# =========================================================

def open_camera(index):
    print(f"Dang mo camera index {index} bang DirectShow...")

    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        print(f"Khong mo duoc camera index {index}.")
        return None

    # Cau hinh camera
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Cho camera khoi dong va kiem tra frame thuc te
    success = False

    for attempt in range(10):
        ret, frame = camera.read()

        if ret and frame is not None and frame.size > 0:
            success = True
            break

        time.sleep(0.3)

    if not success:
        print(f"Camera index {index} mo duoc nhung khong doc duoc frame.")
        camera.release()
        return None

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Camera mo thanh cong.")
    print(f"Backend: DirectShow")
    print(f"Resolution thuc te: {actual_width} x {actual_height}")

    return camera


print("Dang khoi tao camera...")

cap = open_camera(CAMERA_INDEX)

if cap is None:
    raise RuntimeError(
        f"Khong the khoi tao camera index {CAMERA_INDEX}. "
        "Hay kiem tra camera, quyen truy cap hoac thu index 0."
    )


# =========================================================
# 4. OCR + CHUAN HOA BIEN SO
# =========================================================

# Vi tri SO (A->4 ...) va vi tri CHU (4->A ...)
TO_DIGIT = {"A": "4", "O": "0", "Q": "0", "D": "0", "I": "1",
            "L": "1", "B": "8", "S": "5", "Z": "2", "G": "6", "T": "7"}
TO_LETTER = {"4": "A", "0": "O", "1": "I", "8": "B",
             "5": "S", "2": "Z", "6": "G", "7": "T"}

# Cac cap hay bi nham: sua thay the re (0.3) thay vi 1.0
CONFUSABLE = {("A", "4"), ("O", "0"), ("I", "1"), ("B", "8"),
              ("S", "5"), ("Z", "2"), ("G", "6")}


def sub_cost(a, b):
    if a == b:
        return 0.0
    if (a, b) in CONFUSABLE or (b, a) in CONFUSABLE:
        return 0.3
    return 1.0


def weighted_dist(a, b):
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = float(i)
    for j in range(1, m + 1):
        d[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + sub_cost(a[i - 1], b[j - 1]),
            )
    return d[n][m]


def best_registry(text):
    """Tra ve (bien so gan nhat, khoang cach). Khong co registry -> ('', 99)."""
    if not REGISTERED_PLATES:
        return "", 99.0
    best = min((weighted_dist(text, p), p) for p in REGISTERED_PLATES)
    return best[1], best[0]


def registry_gap(text):
    """Khoang cach giua bien gan nhat va bien gan thu nhi. Cang lon cang chac chan."""
    if len(REGISTERED_PLATES) < 2:
        return 99.0
    d = sorted(weighted_dist(text, p) for p in REGISTERED_PLATES)
    return d[1] - d[0]


def fix_positions(s):
    """Bien o to VN: DD L DDDDD. Sua theo vi tri."""
    out = []
    for i, ch in enumerate(s):
        if i == 2:
            out.append(TO_LETTER.get(ch, ch))
        else:
            out.append(TO_DIGIT.get(ch, ch))
    return "".join(out)


def normalize_plate(raw):
    """Lam sach OCR. Neu thua ky tu (dau '-' '.' bi doc thanh '1'...),
    thu bo bot ky tu de ra dung 8 ky tu va chon cach gan registry nhat."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())

    if len(s) < 8:
        return s

    if len(s) == 8:
        return fix_positions(s)

    extra = len(s) - 8
    if extra > 4:
        return fix_positions(s[:8])

    best_text, best_score = None, 1e9
    for drop in combinations(range(3, len(s)), extra):
        t = "".join(c for i, c in enumerate(s) if i not in drop)
        t = fix_positions(t)
        # Ky tu bi bo la '1', 'I', 'L' (giong dau - .) thi it bi phat hon
        non_junk = sum(1 for i in drop if s[i] not in "1IL7")
        _, dist = best_registry(t)
        score = (dist if REGISTERED_PLATES else 0.0) + 0.1 * non_junk
        if score < best_score:
            best_text, best_score = t, score

    return best_text


def read_plate_once(plate_crop):
    """Goi EasyOCR mot lan tren mot crop."""
    h, w = plate_crop.shape[:2]
    scale = max(1.0, min(MAX_UPSCALE, TARGET_CROP_WIDTH / max(w, 1)))

    if scale > 1.01:
        plate_crop = cv2.resize(
            plate_crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(
        gray,
        allowlist=OCR_ALLOWLIST,
        detail=1,
        paragraph=False,
        decoder="greedy",
    )

    if not results:
        return "", 0.0, gray

    results.sort(key=lambda item: min(p[0] for p in item[0]))

    parts, confs = [], []
    for item in results:
        text = item[1].strip().upper()
        if text:
            parts.append(text)
            confs.append(float(item[2]))

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return "".join(parts), avg_conf, gray


# =========================================================
# 5. BURST + BO PHIEU
# =========================================================

def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def detect_plate(frame):
    """Chay YOLO, tra ve (crop, annotated_frame, yolo_conf) hoac None."""
    result = model.predict(
        source=frame, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    idx = int(result.boxes.conf.argmax().item())
    box = result.boxes[idx]
    yolo_conf = float(box.conf.item())

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))

    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None

    px = int((x2 - x1) * PADDING_RATIO_X)
    py = int((y2 - y1) * PADDING_RATIO_Y)
    crop = frame[max(0, y1 - py):min(h, y2 + py),
                 max(0, x1 - px):min(w, x2 + px)]

    if crop.size == 0:
        return None

    return crop, result.plot(), yolo_conf


def burst_recognize():
    # bo frame cu trong buffer
    for _ in range(2):
        cap.grab()

    samples = []   # (crop, annotated, yolo_conf)
    first_frame = None

    for _ in range(BURST_FRAMES):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if first_frame is None:
            first_frame = frame.copy()
        det = detect_plate(frame)
        if det is not None:
            samples.append(det)
        time.sleep(BURST_DELAY)

    if first_frame is not None:
        cv2.imwrite(str(OUTPUT_DIR / "captured_frame.jpg"), first_frame)

    if not samples:
        return None

    # Chi OCR cac frame net nhat
    samples.sort(key=lambda s: sharpness(s[0]), reverse=True)
    samples = samples[:BURST_KEEP]

    votes = defaultdict(float)
    raws = []
    all_clean_exact = True     # moi frame deu doc dung ngay tu dau (khong can sua)
    ocr_confs = []

    for i, (crop, _, _) in enumerate(samples):
        raw, conf, ocr_img = read_plate_once(crop)
        raws.append(raw)
        ocr_confs.append(conf)

        if i == 0:
            cv2.imwrite(str(OUTPUT_DIR / "debug_plate_original.jpg"), crop)
            cv2.imwrite(str(OUTPUT_DIR / "debug_plate_ocr.jpg"), ocr_img)

        text = normalize_plate(raw)

        if REGISTERED_PLATES:
            plate, dist = best_registry(text)
            if dist <= MAX_REGISTRY_DIST and registry_gap(text) < MIN_AMBIGUITY_GAP:
                all_clean_exact = False      # mo ho giua 2 xe -> khong bo phieu
            elif dist <= MAX_REGISTRY_DIST:
                votes[plate] += conf * (1.0 - 0.3 * dist)
                if re.sub(r"[^A-Z0-9]", "", raw.upper()) != plate:
                    all_clean_exact = False
        elif len(text) == 8:
            votes[text] += conf
            all_clean_exact = False   # khong co registry de doi chieu

    avg_ocr_conf = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0
    best_crop, best_annotated, best_yolo = samples[0]

    if not votes:
        guess = normalize_plate(raws[0]) if raws else ""
        return {
            "plate": guess, "raws": raws, "share": 0.0,
            "status": "CHUA DANG KY / CAN XAC MINH",
            "frame": best_annotated, "ocr_conf": avg_ocr_conf,
            "yolo_conf": best_yolo, "vehicle": None,
        }

    plate = max(votes, key=votes.get)
    share = votes[plate] / sum(votes.values())

    if share >= MIN_AGREEMENT and all_clean_exact and len(samples) >= 2:
        status = "KHOP CHINH XAC"
    else:
        status = "CAN XAC MINH THEM"

    vehicle = find_vehicle(plate)
    if not vehicle:
        status = "KHONG TIM THAY XE"

    return {
        "plate": plate, "raws": raws, "share": share, "status": status,
        "frame": best_annotated, "ocr_conf": avg_ocr_conf,
        "yolo_conf": best_yolo, "vehicle": vehicle,
    }


# =========================================================
# 6. CHUP VA HIEN THI
# =========================================================

def capture_and_recognize():
    print("\nDang chup burst va nhan dien...")
    result = burst_recognize()

    if result is None:
        print("YOLO khong phat hien bien so.")
        return

    output = result["frame"].copy()
    vehicle = result["vehicle"]

    cv2.putText(output, f"Plate: {result['plate'] or '(none)'}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if vehicle:
        cv2.putText(output, f"Car: {vehicle['car_id']} Driver: {vehicle['driver']}",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    cv2.putText(output, result["status"], (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(output, f"OCR conf: {result['ocr_conf']:.2f}  vote: {result['share']:.2f}",
                (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("KET QUA NHAN DIEN", output)

    filename = OUTPUT_DIR / f"capture_{int(time.time())}.jpg"
    cv2.imwrite(str(filename), output)

    print("\n========== KET QUA ==========")
    print("Bien so:", result["plate"])
    print("OCR goc tung frame:", result["raws"])
    print(f"Dong thuan phieu: {result['share']:.2f}")
    print("YOLO confidence:", round(result["yolo_conf"], 3))
    print("OCR confidence:", round(result["ocr_conf"], 3))
    print("Trang thai:", result["status"])
    if vehicle:
        print("Ma xe:", vehicle["car_id"])
        print("Tai xe:", vehicle["driver"])
    print("Anh ket qua:", filename)
    print("=============================")


# =========================================================
# 7. CAMERA PREVIEW
# =========================================================

print("\nSPACE: Chup burst va nhan dien")
print("Q / ESC: Thoat")

failed_reads = 0
MAX_FAILED_READS = 10

try:
    while True:

        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:

            failed_reads += 1

            if failed_reads < MAX_FAILED_READS:
                time.sleep(0.1)
                continue

            print("\nMat ket noi camera. Dang thu khoi dong lai...")

            cap.release()
            time.sleep(1)

            cap = open_camera(CAMERA_INDEX)

            if cap is None:
                print(
                    "Khong the ket noi lai camera. "
                    "Kiem tra camera hoac thu CAMERA_INDEX = 0."
                )
                break

            failed_reads = 0
            continue

        # Doc frame thanh cong
        failed_reads = 0

        preview = frame

        if frame.shape[1] > 1280:
            preview = cv2.resize(
                frame,
                (
                    1280,
                    int(frame.shape[0] * 1280 / frame.shape[1])
                ),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow("VIN PARKING - CAMERA", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            capture_and_recognize()

        elif key in (ord("q"), ord("Q"), 27):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Da dong camera.")