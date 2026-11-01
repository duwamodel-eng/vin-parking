
from pathlib import Path

import cv2
import easyocr


# =========================================================
# 1. DUONG DAN
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "camera_results"

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"

# Lay toi da 10 anh crop moi nhat
MAX_IMAGES = 10


# =========================================================
# 2. KIEM TRA THU MUC VA ANH
# =========================================================

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay thu muc anh: {IMAGE_DIR}"
    )

crop_files = sorted(
    IMAGE_DIR.glob("crop_*.jpg"),
    key=lambda path: path.stat().st_mtime
)

crop_files = crop_files[-MAX_IMAGES:]

if not crop_files:
    raise FileNotFoundError(
        f"Khong tim thay crop_*.jpg trong: {IMAGE_DIR}"
    )

print("Thu muc anh:", IMAGE_DIR)
print("So anh se kiem tra:", len(crop_files))


# =========================================================
# 3. KHOI TAO EASYOCR
# =========================================================

print("\nDang khoi dong EasyOCR...")
reader = easyocr.Reader(["en"], gpu=True)
print("EasyOCR da san sang.")


# =========================================================
# 4. HAM CHUAN HOA KET QUA
# =========================================================

def read_plate(image):
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist=OCR_ALLOWLIST
    )

    results.sort(
        key=lambda item: min(point[0] for point in item[0])
    )

    text = "".join(
        item[1].replace(" ", "").upper()
        for item in results
    )

    confidence = (
        sum(float(item[2]) for item in results) / len(results)
        if results else 0.0
    )

    return text, confidence


# =========================================================
# 5. KIEM TRA CAC PHUONG PHAP TIEN XU LY
# =========================================================

for image_path in crop_files:

    original = cv2.imread(str(image_path))

    if original is None:
        print("\nKhong doc duoc:", image_path.name)
        continue

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    # Tang tuong phan cuc bo
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    # Lam net nhe
    blur = cv2.GaussianBlur(
        clahe,
        (0, 0),
        1.0
    )

    sharpen = cv2.addWeighted(
        clahe,
        1.5,
        blur,
        -0.5,
        0
    )

    # Nhi phan Otsu
    _, otsu = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Phong to anh truoc khi OCR
    variants = {
        "original_3x": cv2.resize(
            original, None,
            fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        ),

        "gray_3x": cv2.resize(
            gray, None,
            fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        ),

        "clahe_3x": cv2.resize(
            clahe, None,
            fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        ),

        "sharpen_3x": cv2.resize(
            sharpen, None,
            fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        ),

        "otsu_3x": cv2.resize(
            otsu, None,
            fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        )
    }

    print("\n" + "=" * 60)
    print("ANH:", image_path.name)
    print("Kich thuoc crop:", original.shape)
    print("=" * 60)

    for name, variant in variants.items():

        try:
            text, confidence = read_plate(variant)

            print(
                f"{name:12s} | "
                f"plate={text or '---':15s} | "
                f"OCR={confidence:.2f}"
            )

        except Exception as error:
            print(f"{name}: LOI OCR - {error}")

print("\nDA KIEM TRA XONG CAC ANH.")
