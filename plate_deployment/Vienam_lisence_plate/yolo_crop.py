
from ultralytics import YOLO
import cv2

# Model YOLO
model = YOLO(
    r"D:\Vienam_lisence_plate\runs\detect\train\weights\best.pt"
)

# Ảnh xe
image_path = r"D:\Vienam_lisence_plate\test_plates\car_test.jpg"

image = cv2.imread(image_path)

# YOLO detect
results = model(image, conf=0.25)

for i, result in enumerate(results):

    boxes = result.boxes

    if len(boxes) == 0:
        print("Khong tim thay bien so.")
        continue

    # Lấy box đầu tiên
    box = boxes[0]

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    confidence = float(box.conf[0])

    print("License plate detected!")
    print(f"Confidence: {confidence:.2f}")
    print(f"Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Crop biển số
    plate_crop = image[y1:y2, x1:x2]

    # Lưu crop
    output_path = r"D:\Vienam_lisence_plate\test_plates\plate_crop.jpg"

    cv2.imwrite(output_path, plate_crop)

    print(f"Saved: {output_path}")
    