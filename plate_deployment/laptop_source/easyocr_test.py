import easyocr
import cv2
import os

# Khởi tạo EasyOCR bằng GPU RTX 4050
reader = easyocr.Reader(['en'], gpu=True)

folder = r"D:\Vienam_lisence_plate\test_plates"

for filename in sorted(os.listdir(folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(folder, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"Khong doc duoc: {filename}")
        continue

    results = reader.readtext(image)

    print("\n==============================")
    print("FILE:", filename)

    if not results:
        print("Khong tim thay text")
        continue

    for box, text, confidence in results:
        print(f"Text       : {text}")
        print(f"Confidence: {confidence:.2f}")