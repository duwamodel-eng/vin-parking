
import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Khong mo duoc camera")
    raise SystemExit

print("Dang kiem tra camera. Nhan Q de thoat.")

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Khong doc duoc frame")
            continue

        cv2.imshow("CAMERA CHECK", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
