import easyocr
import cv2

print("Dang khoi dong EasyOCR...")

# EasyOCR dung GPU RTX 4050
reader = easyocr.Reader(['en'], gpu=True)

# Anh bien so da duoc YOLO crop
image_path = r"D:\Vienam_lisence_plate\test_plates\plate_crop.jpg"

print("Dang doc:", image_path)

image = cv2.imread(image_path)

if image is None:
    print("KHONG DOC DUOC plate_crop.jpg")
    exit()

print("Dang OCR...")

results = reader.readtext(image)

print("\n")
print("OCR RESULT")

if not results:
    print("Khong tim thay text")
else:
    for box, text, confidence in results:
        print(f"Text       : {text}")
        print(f"Confidence: {confidence:.2f}")

print("")