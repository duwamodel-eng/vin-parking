
import cv2
import time
from pathlib import Path

CAMERA = "/dev/video0"  # ENTRY
OUTPUT = Path(__file__).resolve().parent / "camera_entry_test.jpg"

def main():
    print("Opening ENTRY camera:", CAMERA)

    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # Yêu cầu camera xuất hình MJPG 1280x720
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Waiting for camera to stabilize...")

    frame = None

    # Bỏ qua các frame đầu để camera ổn định
    for i in range(30):
        ret, current = cap.read()

        if ret and current is not None:
            frame = current

        time.sleep(0.03)

    cap.release()

    if frame is None:
        print("ERROR: Camera opened but no valid frame received.")
        return

    height, width = frame.shape[:2]

    if not cv2.imwrite(str(OUTPUT), frame):
        print("ERROR: Cannot save image.")
        return

    print("CAMERA TEST SUCCESS")
    print("Device:", CAMERA)
    print("Actual resolution:", width, "x", height)
    print("Image saved:", OUTPUT)

if __name__ == "__main__":
    main()
