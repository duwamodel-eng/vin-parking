
import base64
import json
import re
import secrets
import sqlite3
import time

import cv2
import face_recognition
import numpy as np
from flask import Blueprint, current_app, jsonify, request, session

from services.app_context import (
    PENDING_FACES,
    get_db,
    json_error,
    normalize_phone,
    valid_full_name,
)
from services.auth_utils import login_required


driver_bp = Blueprint("driver_api", __name__)

POSES = ("FRONT", "LEFT", "RIGHT")
MIN_SAMPLES_PER_POSE = 1
MAX_SAMPLES_PER_POSE = 1 
FACE_TOKEN_TTL = 300

# Ngưỡng phát hiện khuôn mặt trùng khi đăng ký.
# Đây là ngưỡng ban đầu dựa trên dữ liệu dương tính hiện có,
# cần tiếp tục kiểm thử với người khác để đánh giá false positive.
FACE_DUPLICATE_THRESHOLD = 0.42
FACE_DUPLICATE_MEDIAN_LIMIT = 0.30
FACE_DUPLICATE_MIN_RATIO = 0.70
FACE_DUPLICATE_NEIGHBORS = 3


def cleanup_pending_faces():
    """Xóa dữ liệu khuôn mặt tạm đã hết hạn."""
    now = time.time()

    expired = [
        token
        for token, item in list(PENDING_FACES.items())
        if now - item.get("created_at", 0) > FACE_TOKEN_TTL
    ]

    for token in expired:
        PENDING_FACES.pop(token, None)


def decode_face_image(image_data, max_size=1280):
    """Giải mã một ảnh JPEG và trả về embedding 128 chiều."""
    if not isinstance(image_data, str):
        return None

    match = re.fullmatch(
        r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)",
        image_data,
    )
    if not match:
        return None

    try:
        raw = base64.b64decode(match.group(1), validate=True)

        if not raw or len(raw) > 5 * 1024 * 1024:
            return None

        array = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if bgr is None:
            return None

        height, width = bgr.shape[:2]

        if min(height, width) < 100:
            return None

        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            bgr = cv2.resize(
                bgr,
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                ),
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(
            rgb,
            model="hog",
        )

        if len(locations) != 1:
            return None

        found = face_recognition.face_encodings(
            rgb,
            known_face_locations=locations,
        )

        if len(found) != 1:
            return None

        encoding = np.asarray(found[0], dtype=np.float64)

        if encoding.shape != (128,) or not np.isfinite(encoding).all():
            return None

        return encoding

    except (ValueError, TypeError, cv2.error):
        return None


def load_valid_embedding(encoding_json):
    """Đọc và kiểm tra embedding JSON."""
    try:
        encoding = np.asarray(
            json.loads(encoding_json),
            dtype=np.float64,
        )

        if encoding.shape != (128,) or not np.isfinite(encoding).all():
            return None

        return encoding

    except (TypeError, ValueError, json.JSONDecodeError):
        return None

def load_registration_face_profiles(conn):
    """
    Tải toàn bộ hồ sơ khuôn mặt đã đăng ký.

    Ưu tiên các embedding riêng lẻ trong face_embedding_samples.
    Chỉ dùng encoding_json của face_encodings khi hồ sơ chưa có
    bất kỳ mẫu riêng lẻ nào.

    Bao gồm cả tài xế INACTIVE để không cho đăng ký lại
    cùng một khuôn mặt dưới mã tài xế khác.
    """

    rows = conn.execute("""
        SELECT driver_id, encoding_json
        FROM face_encodings
        ORDER BY driver_id
    """).fetchall()

    profiles = {}

    for row in rows:
        driver_id = row["driver_id"]

        sample_rows = conn.execute("""
            SELECT id, pose, encoding_json
            FROM face_embedding_samples
            WHERE driver_id = ?
            ORDER BY id
        """, (driver_id,)).fetchall()

        embeddings = []

        if sample_rows:
            for sample in sample_rows:
                vector = load_valid_embedding(
                    sample["encoding_json"]
                )

                if vector is None:
                    raise ValueError(
                        f"Embedding mẫu không hợp lệ: "
                        f"driver_id={driver_id}, "
                        f"sample_id={sample['id']}"
                    )

                embeddings.append(vector)

        else:
            # Hồ sơ cũ chưa có mẫu riêng lẻ.
            vector = load_valid_embedding(row["encoding_json"])

            if vector is None:
                raise ValueError(
                    f"Embedding đại diện không hợp lệ: "
                    f"driver_id={driver_id}"
                )

            embeddings.append(vector)

        if not embeddings:
            raise ValueError(
                f"Hồ sơ {driver_id} không có embedding hợp lệ."
            )

        profiles[driver_id] = np.stack(embeddings)

    return profiles


def find_duplicate_faces(new_encodings, existing_profiles):
    """
    So sánh toàn bộ embedding mới với từng hồ sơ cũ.

    Mỗi embedding mới được so với tối đa 3 mẫu gần nhất
    trong một hồ sơ cũ.

    Chỉ đánh dấu trùng khi đồng thời:
    - Median điểm của các mẫu mới < 0.30.
    - Ít nhất 70% mẫu mới có điểm < 0.42.
    """

    new_array = np.asarray(
        new_encodings,
        dtype=np.float64
    )

    if (
        new_array.ndim != 2
        or new_array.shape[1] != 128
        or len(new_array) == 0
        or not np.isfinite(new_array).all()
    ):
        raise ValueError(
            "Các embedding mới không hợp lệ."
        )

    duplicates = []

    for driver_id, known_encodings in existing_profiles.items():
        known_array = np.asarray(
            known_encodings,
            dtype=np.float64
        )

        if (
            known_array.ndim != 2
            or known_array.shape[1] != 128
            or len(known_array) == 0
            or not np.isfinite(known_array).all()
        ):
            raise ValueError(
                f"Embedding của tài xế {driver_id} không hợp lệ."
            )

        # Khoảng cách giữa từng mẫu mới và từng mẫu cũ.
        distances = np.linalg.norm(
            new_array[:, None, :] - known_array[None, :, :],
            axis=2
        )

        nearest_count = min(
            FACE_DUPLICATE_NEIGHBORS,
            len(known_array)
        )

        nearest_distances = np.sort(
            distances,
            axis=1
        )[:, :nearest_count]

        # Điểm của từng mẫu mới đối với hồ sơ cũ.
        sample_scores = np.median(
            nearest_distances,
            axis=1
        )

        median_score = float(
            np.median(sample_scores)
        )

        similar_ratio = float(
            np.mean(
                sample_scores < FACE_DUPLICATE_THRESHOLD
            )
        )

        current_app.logger.info(
            "[FACE DUPLICATE CHECK] "
            "driver=%s median=%.5f ratio=%.3f samples=%d",
            driver_id,
            median_score,
            similar_ratio,
            len(sample_scores)
        )

        if (
            median_score < FACE_DUPLICATE_MEDIAN_LIMIT
            and similar_ratio >= FACE_DUPLICATE_MIN_RATIO
        ):
            duplicates.append({
                "driver_id": driver_id,
                "median_distance": median_score,
                "similar_ratio": similar_ratio
            })

    return duplicates

@driver_bp.route("/api/drivers")
@login_required
def api_drivers():
    conn = get_db()

    try:
        rows = conn.execute("""
            SELECT
                v.plate_number,
                v.driver_id,
                d.full_name,
                d.phone,
                d.status,
                d.created_at,
                r.uid,
                CASE
                    WHEN f.driver_id IS NOT NULL THEN 1
                    ELSE 0
                END AS face_registered
            FROM vehicles AS v
            LEFT JOIN drivers AS d
                ON d.driver_id = v.driver_id
            LEFT JOIN rfid_cards AS r
                ON r.driver_id = d.driver_id
            LEFT JOIN face_encodings AS f
                ON f.driver_id = d.driver_id
            ORDER BY v.plate_number
        """).fetchall()

        drivers = [{
            "plate_number": row["plate_number"],
            "driver_id": row["driver_id"],
            "full_name": row["full_name"],
            "phone": row["phone"],
            "status": row["status"],
            "created_at": row["created_at"],
            "uid": row["uid"],
            "face_registered": bool(row["face_registered"]),
            "registered": row["driver_id"] is not None,
        } for row in rows]

        available_plates = [
            row["plate_number"]
            for row in rows
            if row["driver_id"] is None
        ]

        registered_total = sum(
            1 for row in rows
            if row["driver_id"] is not None
        )

        return jsonify({
            "success": True,
            "drivers": drivers,
            "total": len(drivers),
            "registered_total": registered_total,
            "available_plates": available_plates,
        })

    except sqlite3.Error:
        current_app.logger.exception(
            "Lỗi truy vấn danh sách tài xế và biển số"
        )
        return json_error("Không thể tải danh sách tài xế.", 500)

    finally:
        conn.close()


@driver_bp.route("/api/face/encode", methods=["POST"])
@login_required
def encode_driver_face():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return json_error("Dữ liệu ảnh không hợp lệ.")

    poses_data = data.get("poses")

    if not isinstance(poses_data, dict):
        return json_error(
            "Dữ liệu phải có đủ 3 nhóm ảnh FRONT, LEFT và RIGHT."
        )

    pose_samples = {}
    all_encodings = []

    for pose in POSES:
        images = poses_data.get(pose)

        if (
            not isinstance(images, list)
            or not MIN_SAMPLES_PER_POSE <= len(images) <= MAX_SAMPLES_PER_POSE
        ):
            return json_error(
                f"Góc {pose} cần từ {MIN_SAMPLES_PER_POSE} "
                f"đến {MAX_SAMPLES_PER_POSE} ảnh."
            )

        valid_encodings = []

        for image_data in images:
            encoding = decode_face_image(image_data)

            if encoding is not None:
                valid_encodings.append(encoding)

        if len(valid_encodings) < MIN_SAMPLES_PER_POSE:
            return json_error(
                f"Góc {pose} chỉ nhận được {len(valid_encodings)} ảnh hợp lệ. "
                "Hãy chụp lại góc này, đảm bảo đủ sáng và chỉ có một khuôn mặt."
            )

        pose_samples[pose] = [
            encoding.tolist()
            for encoding in valid_encodings
        ]

        all_encodings.extend(valid_encodings)

    # Kiểm tra chất lượng tổng thể, không lấy trung bình thay thế các mẫu.
    encoding_array = np.stack(all_encodings)

    distance_matrix = np.linalg.norm(
        encoding_array[:, None, :] - encoding_array[None, :, :],
        axis=2,
    )

    pairwise_distances = distance_matrix[
        np.triu_indices(len(all_encodings), k=1)
    ]

    current_app.logger.info(
        "[FACE ENROLLMENT QUALITY] total=%d "
        "min=%.6f median=%.6f max=%.6f",
        len(all_encodings),
        float(np.min(pairwise_distances)),
        float(np.median(pairwise_distances)),
        float(np.max(pairwise_distances)),
    )

    # Vector đại diện chỉ để tương thích với bảng face_encodings cũ.
    representative = np.median(encoding_array, axis=0)

    if not np.isfinite(representative).all():
        return json_error("Không tạo được dữ liệu khuôn mặt hợp lệ.")

    # Kiểm tra trùng với toàn bộ embedding đã đăng ký.
    check_conn = None

    try:
        check_conn = get_db()

        existing_profiles = load_registration_face_profiles(
            check_conn
        )

        duplicate_faces = find_duplicate_faces(
            encoding_array,
            existing_profiles
        )

        if duplicate_faces:
            matched_ids = ", ".join(
                item["driver_id"]
                for item in duplicate_faces
            )

            current_app.logger.warning(
                "[FACE ENROLLMENT BLOCKED] duplicates=%s",
                duplicate_faces
            )

            return json_error(
                "Khuôn mặt có dấu hiệu đã được đăng ký "
                f"trong hồ sơ: {matched_ids}. "
                "Không thể đăng ký tài xế mới bằng khuôn mặt này.",
                409
            )

    except (sqlite3.Error, ValueError):
        current_app.logger.exception(
            "Không thể kiểm tra trùng khuôn mặt khi đăng ký."
        )

        return json_error(
            "Không thể kiểm tra dữ liệu khuôn mặt hiện có. "
            "Hệ thống tạm dừng đăng ký để bảo đảm an toàn.",
            503
        )

    finally:
        if check_conn is not None:
            check_conn.close()  

    cleanup_pending_faces()

    token = secrets.token_urlsafe(32)

    PENDING_FACES[token] = {
        "encoding": representative.tolist(),
        "pose_samples": pose_samples,
        "samples": len(all_encodings),
        "created_at": time.time(),
        "user_id": session["user_id"],
    }

    return jsonify({
        "success": True,
        "face_token": token,
        "samples": len(all_encodings),
        "pose_counts": {
            pose: len(pose_samples[pose])
            for pose in POSES
        },
        "message": (
            f"Đã xử lý {len(all_encodings)} mẫu khuôn mặt "
            "ở 3 góc FRONT, LEFT và RIGHT."
        ),
    })

# ////////////////////////////////////////////////////////////////////////
@driver_bp.route("/api/drivers/register", methods=["POST"])
@login_required
def register_driver():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return json_error("Dữ liệu gửi lên không hợp lệ.")

    driver_id = str(data.get("driver_id", "")).strip().upper()
    plate_number = str(data.get("plate_number", "")).strip().upper()
    full_name = valid_full_name(data.get("full_name"))
    phone = normalize_phone(data.get("phone"))
    uid = str(data.get("uid", "")).strip().upper()
    face_token = str(data.get("face_token", "")).strip()

    if not re.fullmatch(r"[A-Z0-9_-]{2,30}", driver_id):
        return json_error(
            "Mã tài xế phải có 2–30 ký tự chữ, số, _ hoặc -."
        )

    if not re.fullmatch(r"\d{2}[A-Z]\d{5}", plate_number):
        return json_error("Biển số xe không hợp lệ.")

    if full_name is None:
        return json_error(
            "Họ tên không hợp lệ; không để trống, không chứa số "
            "và tối đa 100 ký tự."
        )

    if phone is None:
        return json_error(
            "Số điện thoại phải có 10 chữ số, bắt đầu bằng 0."
        )

    if not re.fullmatch(r"(?:[0-9A-F]{2}:)+[0-9A-F]{2}", uid):
        return json_error("UID RFID không đúng định dạng.")

    cleanup_pending_faces()

    face = PENDING_FACES.get(face_token)

    if (
        not face
        or face.get("user_id") != session["user_id"]
        or time.time() - face.get("created_at", 0) > FACE_TOKEN_TTL
    ):
        return json_error(
            "Khuôn mặt chưa được xử lý hoặc token đã hết hạn. Hãy chụp lại."
        )

    encoding = face.get("encoding")
    pose_samples = face.get("pose_samples")

    if (
        not isinstance(encoding, list)
        or len(encoding) != 128
        or not np.isfinite(encoding).all()
    ):
        return json_error("Dữ liệu khuôn mặt đại diện không hợp lệ.")

    if not isinstance(pose_samples, dict):
        return json_error("Không tìm thấy các mẫu khuôn mặt theo góc.")

    # Kiểm tra lại toàn bộ mẫu trước khi bắt đầu transaction.
    validated_samples = []

    for pose in POSES:
        samples = pose_samples.get(pose)

        if (
            not isinstance(samples, list)
            or not MIN_SAMPLES_PER_POSE <= len(samples) <= MAX_SAMPLES_PER_POSE
        ):
            return json_error(f"Dữ liệu mẫu góc {pose} không hợp lệ.")

        for sample in samples:
            vector = np.asarray(sample, dtype=np.float64)

            if vector.shape != (128,) or not np.isfinite(vector).all():
                return json_error(
                    f"Embedding góc {pose} không hợp lệ."
                )

            validated_samples.append((pose, vector.tolist()))

    conn = get_db()

    try:
        conn.execute("BEGIN IMMEDIATE")

        # Kiểm tra trùng lần hai bên trong transaction.
        # Không cho phép yêu cầu khác chen vào giữa kiểm tra và INSERT.
        existing_profiles = load_registration_face_profiles(
            conn
        )

        new_samples_array = np.stack([
            np.asarray(sample, dtype=np.float64)
            for pose, sample in validated_samples
        ])

        duplicate_faces = find_duplicate_faces(
            new_samples_array,
            existing_profiles
        )

        if duplicate_faces:
            matched_ids = ", ".join(
                item["driver_id"]
                for item in duplicate_faces
            )

            current_app.logger.warning(
                "[FACE REGISTER BLOCKED] "
                "new_driver=%s duplicates=%s",
                driver_id,
                duplicate_faces
            )

            conn.rollback()

            # Hủy token để không thể gửi lại cùng bộ ảnh.
            PENDING_FACES.pop(face_token, None)

            return json_error(
                "Khuôn mặt đã có dấu hiệu trùng với hồ sơ: "
                f"{matched_ids}. "
                "Không thể đăng ký tài xế mới.",
                409
            )

        vehicle = conn.execute("""
            SELECT driver_id
            FROM vehicles
            WHERE plate_number = ?
        """, (plate_number,)).fetchone()

        if vehicle is None:
            conn.rollback()
            return json_error(
                "Biển số không tồn tại trong danh sách xe cố định.",
                404,
            )

        if vehicle["driver_id"] is not None:
            conn.rollback()
            return json_error(
                "Biển số này đã được gán cho tài xế khác.",
                409,
            )

        uid_exists = conn.execute("""
            SELECT driver_id
            FROM rfid_cards
            WHERE UPPER(uid) = ?
            LIMIT 1
        """, (uid,)).fetchone()

        if uid_exists:
            conn.rollback()
            return json_error(
                f"UID RFID đã được đăng ký cho tài xế "
                f"{uid_exists['driver_id']}. Vui lòng dùng thẻ khác.",
                409,
            )

        # Ghi hồ sơ, RFID, vector tương thích và các mẫu theo góc
        # trong cùng một transaction.
        conn.execute("""
            INSERT INTO drivers (driver_id, full_name, phone, status)
            VALUES (?, ?, ?, 'ACTIVE')
        """, (driver_id, full_name, phone))

        conn.execute("""
            INSERT INTO rfid_cards (uid, driver_id)
            VALUES (?, ?)
        """, (uid, driver_id))

        conn.execute("""
            INSERT INTO face_encodings
                (driver_id, encoding_json, samples, updated_at)
            VALUES (?, ?, ?, datetime('now', 'localtime'))
        """, (
            driver_id,
            json.dumps(encoding),
            len(validated_samples),
        ))

        conn.executemany("""
            INSERT INTO face_embedding_samples
                (driver_id, pose, encoding_json)
            VALUES (?, ?, ?)
        """, [
            (driver_id, pose, json.dumps(sample))
            for pose, sample in validated_samples
        ])

        cur = conn.execute("""
            UPDATE vehicles
            SET driver_id = ?
            WHERE plate_number = ?
              AND driver_id IS NULL
        """, (driver_id, plate_number))

        if cur.rowcount != 1:
            conn.rollback()
            return json_error(
                "Không thể gán biển số. Vui lòng tải lại danh sách.",
                409,
            )

        conn.commit()

        PENDING_FACES.pop(face_token, None)

        return jsonify({
            "success": True,
            "message": "Đăng ký tài xế thành công.",
            "face_samples": len(validated_samples),
        }), 201

    except sqlite3.IntegrityError:
        conn.rollback()
        return json_error(
            "Mã tài xế, UID hoặc hồ sơ khuôn mặt đã tồn tại.",
            409,
        )

    except ValueError:
        conn.rollback()

        current_app.logger.exception(
            "Embedding không hợp lệ khi đăng ký tài xế."
        )

        return json_error(
            "Không thể kiểm tra dữ liệu khuôn mặt hiện có. "
            "Hồ sơ chưa được lưu.",
            503
        )
    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi lưu hồ sơ tài xế")
        return json_error(
            "Lỗi cơ sở dữ liệu. Hồ sơ chưa được lưu.",
            500,
        )

    finally:
        conn.close()


@driver_bp.route("/api/drivers/<driver_id>", methods=["PUT"])
@login_required
def update_driver(driver_id):
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return json_error("Dữ liệu JSON không hợp lệ.")

    full_name = valid_full_name(data.get("full_name"))
    phone = normalize_phone(data.get("phone"))
    status_value = str(data.get("status", "")).strip().upper()

    if full_name is None:
        return json_error("Họ tên không hợp lệ.")

    if phone is None:
        return json_error(
            "Số điện thoại phải có 10 chữ số, bắt đầu bằng 0."
        )

    if status_value not in ("ACTIVE", "INACTIVE"):
        return json_error(
            "Trạng thái chỉ được là ACTIVE hoặc INACTIVE."
        )

    conn = get_db()

    try:
        conn.execute("BEGIN IMMEDIATE")

        cur = conn.execute("""
            UPDATE drivers
            SET full_name = ?, phone = ?, status = ?
            WHERE driver_id = ?
        """, (full_name, phone, status_value, driver_id))

        if cur.rowcount == 0:
            conn.rollback()
            return json_error("Không tìm thấy tài xế.", 404)

        conn.commit()

        return jsonify({
            "success": True,
            "message": "Cập nhật tài xế thành công.",
        })

    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi cập nhật tài xế")
        return json_error("Không thể cập nhật tài xế.", 500)

    finally:
        conn.close()


@driver_bp.route("/api/drivers/<driver_id>", methods=["DELETE"])
@login_required
def delete_driver(driver_id):
    conn = get_db()

    try:
        conn.execute("BEGIN IMMEDIATE")

        exists = conn.execute("""
            SELECT 1
            FROM drivers
            WHERE driver_id = ?
        """, (driver_id,)).fetchone()

        if exists is None:
            conn.rollback()
            return json_error("Không tìm thấy tài xế.", 404)

        # Xóa các mẫu con trước để không phụ thuộc vào trạng thái
        # PRAGMA foreign_keys của kết nối.
        conn.execute("""
            DELETE FROM face_embedding_samples
            WHERE driver_id = ?
        """, (driver_id,))

        conn.execute("""
            DELETE FROM face_encodings
            WHERE driver_id = ?
        """, (driver_id,))

        conn.execute("""
            DELETE FROM rfid_cards
            WHERE driver_id = ?
        """, (driver_id,))

        # Không xóa parking_history.
        # FK vehicles.driver_id ON DELETE SET NULL sẽ giải phóng biển số.
        conn.execute("""
            DELETE FROM drivers
            WHERE driver_id = ?
        """, (driver_id,))

        conn.commit()

        return jsonify({
            "success": True,
            "message": (
                "Đã xóa hồ sơ tài xế và mẫu khuôn mặt. "
                "Biển số được giữ lại và chuyển về trạng thái chưa đăng ký."
            ),
        })

    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi xóa tài xế")
        return json_error("Không thể xóa tài xế.", 500)

    finally:
        conn.close()
