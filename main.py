import os
import time
import cv2
import face_recognition

# ==========================================
# 1. CAU HINH
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "face")

CAMERA_INDEX = 0
MATCH_TOLERANCE = 0.48

# Camera 640x480, uu tien MJPG de giam tai USB
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Xu ly moi khung hinh sau mot khoang nghi nho
PROCESS_INTERVAL = 0.2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

known_encodings = []
known_names = []

if not os.path.isdir(FACES_DIR):
    raise SystemExit(f"[LOI] Khong tim thay thu muc: {FACES_DIR}")


# ==========================================
# 2. NAP DU LIEU KHUON MAT
# ==========================================
print("[INFO] Dang nap du lieu khuon mat...")

for person_name in sorted(os.listdir(FACES_DIR)):
    person_dir = os.path.join(FACES_DIR, person_name)

    if not os.path.isdir(person_dir):
        continue

    person_count = 0
    person_skipped = 0

    for img_file in sorted(os.listdir(person_dir)):
        if not img_file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        img_path = os.path.join(person_dir, img_file)

        try:
            image = face_recognition.load_image_file(img_path)

            # Phat hien khuon mat mot lan
            locations = face_recognition.face_locations(
                image,
                number_of_times_to_upsample=1,
                model="hog"
            )

            if not locations:
                print(f"[BO QUA] Khong tim thay mat: {img_file}")
                person_skipped += 1
                continue

            # Dung lai vi tri mat da phat hien
            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=locations
            )

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                person_count += 1
            else:
                person_skipped += 1

        except Exception as e:
            print(f"[LOI] {img_file}: {e}")
            person_skipped += 1

    print(
        f"[INFO] {person_name}: "
        f"{person_count} anh hop le, "
        f"{person_skipped} anh bo qua"
    )

print(f"\n[INFO] Tong so vector khuon mat: {len(known_encodings)}")

if not known_encodings:
    raise SystemExit(
        "[LOI] Khong co vector khuon mat. "
        "Hay kiem tra thu muc face."
    )


# ==========================================
# 3. KHOI DONG WEBCAM
# ==========================================
print(f"[INFO] Dang mo webcam /dev/video{CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    cap.release()
    raise SystemExit(
        f"[LOI] Khong mo duoc webcam /dev/video{CAMERA_INDEX}."
    )

# Yeu cau webcam su dung MJPG va kich thuoc 640x480
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Doc mot khung hinh de xac nhan camera hoat dong
ret, frame = cap.read()

if not ret or frame is None:
    cap.release()
    raise SystemExit(
        "[LOI] Mo duoc webcam nhung khong doc duoc hinh anh."
    )

print("[INFO] Webcam hoat dong.")
print("[INFO] Dang nhan dien qua SSH; nhan Ctrl+C de dung.")
print("[INFO] Ket qua se duoc in ra terminal.")


# ==========================================
# 4. NHAN DIEN KHUON MAT
# ==========================================
last_process_time = 0
last_result = None
last_result_time = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[CANH BAO] Khong doc duoc khung hinh tu webcam.")
            time.sleep(0.5)
            continue

        now = time.monotonic()

        # Gioi han tan suat xu ly de giam tai CPU
        if now - last_process_time < PROCESS_INTERVAL:
            continue

        last_process_time = now

        # Thu nho khung hinh truoc khi phat hien khuon mat
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        rgb_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_frame,
            model="hog",
            number_of_times_to_upsample=0
        )

        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=face_locations
        )

        if not face_encodings:
            # Chi thong bao khi trang thai thay doi
            if last_result != "NO_FACE":
                print("[KET QUA] Chua phat hien khuon mat.")
                last_result = "NO_FACE"
                last_result_time = now

            continue

        # Xu ly tung khuon mat trong khung hinh
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_match_idx = distances.argmin()
            best_distance = float(distances[best_match_idx])

            if best_distance < MATCH_TOLERANCE:
                name = known_names[best_match_idx]
                result = f"VALID:{name}"

                # Khong lap lai cung mot thong bao lien tuc
                if result != last_result or now - last_result_time >= 3:
                    print(
                        f"[NHAN DIEN] Hop le: {name} "
                        f"(distance={best_distance:.3f})"
                    )
                    last_result = result
                    last_result_time = now

            else:
                result = "UNKNOWN"

                if result != last_result or now - last_result_time >= 3:
                    print(
                        "[NHAN DIEN] Nguoi la - khong duoc xac nhan "
                        f"(distance={best_distance:.3f})"
                    )
                    last_result = result
                    last_result_time = now

except KeyboardInterrupt:
    print("\n[INFO] Nguoi dung da dung chuong trinh.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Da dong webcam.")
