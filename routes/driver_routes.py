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


def cleanup_pending_faces():
    """Xóa dữ liệu khuôn mặt tạm đã hết hạn."""
    now = time.time()
    expired = [
        token
        for token, item in PENDING_FACES.items()
        if now - item["created_at"] > 300
    ]
    for token in expired:
        PENDING_FACES.pop(token, None)


@driver_bp.route("/api/drivers")
@login_required
def api_drivers():
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT d.driver_id, d.full_name, d.phone, d.status, d.created_at,
                   r.uid,
                   CASE WHEN f.driver_id IS NOT NULL THEN 1 ELSE 0 END AS face_registered
            FROM drivers AS d
            LEFT JOIN rfid_cards AS r ON r.driver_id = d.driver_id
            LEFT JOIN face_encodings AS f ON f.driver_id = d.driver_id
            ORDER BY d.created_at DESC, d.driver_id
        """).fetchall()

        drivers = [{
            "driver_id": r["driver_id"],
            "full_name": r["full_name"],
            "phone": r["phone"],
            "status": r["status"],
            "created_at": r["created_at"],
            "uid": r["uid"],
            "face_registered": bool(r["face_registered"]),
        } for r in rows]

        return jsonify({
            "success": True,
            "drivers": drivers,
            "total": len(drivers),
        })
    except sqlite3.Error:
        current_app.logger.exception("Lỗi truy vấn danh sách tài xế")
        return json_error("Không thể tải danh sách tài xế.", 500)
    finally:
        conn.close()


@driver_bp.route("/api/face/encode", methods=["POST"])
@login_required
def encode_driver_face():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("Dữ liệu ảnh không hợp lệ.")

    image_data = data.get("image", "")
    if not isinstance(image_data, str):
        return json_error("Ảnh không đúng định dạng JPEG.")

    match = re.fullmatch(
        r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)",
        image_data,
    )
    if not match:
        return json_error("Ảnh không đúng định dạng JPEG.")

    try:
        raw = base64.b64decode(match.group(1), validate=True)
        if not raw or len(raw) > 5 * 1024 * 1024:
            return json_error("Ảnh rỗng hoặc vượt quá 5 MB.")

        image_array = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if bgr is None:
            return json_error("Không đọc được ảnh.")

        height, width = bgr.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            bgr = cv2.resize(
                bgr,
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                ),
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if len(locations) != 1:
            return json_error(
                f"Cần đúng một khuôn mặt trong ảnh. "
                f"Hiện phát hiện: {len(locations)}."
            )

        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=locations,
        )
        if len(encodings) != 1 or not np.isfinite(encodings[0]).all():
            return json_error("Không tạo được dữ liệu khuôn mặt hợp lệ.")

        cleanup_pending_faces()
        token = secrets.token_urlsafe(32)
        PENDING_FACES[token] = {
            "encoding": encodings[0].tolist(),
            "created_at": time.time(),
            "user_id": session["user_id"],
        }

        return jsonify({
            "success": True,
            "face_token": token,
            "message": "Đã tạo dữ liệu khuôn mặt.",
        })
    except (ValueError, cv2.error):
        return json_error("Ảnh không hợp lệ hoặc không thể xử lý.")


@driver_bp.route("/api/drivers/register", methods=["POST"])
@login_required
def register_driver():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("Dữ liệu gửi lên không hợp lệ.")

    driver_id = str(data.get("driver_id", "")).strip().upper()
    full_name = valid_full_name(data.get("full_name"))
    phone = normalize_phone(data.get("phone"))
    uid = str(data.get("uid", "")).strip().upper()
    face_token = str(data.get("face_token", "")).strip()

    if not re.fullmatch(r"[A-Z0-9_-]{2,30}", driver_id):
        return json_error("Mã tài xế phải có 2–30 ký tự chữ, số, _ hoặc -.")
    if full_name is None:
        return json_error(
            "Họ tên không hợp lệ; không để trống, không chứa số và tối đa 100 ký tự."
        )
    if phone is None:
        return json_error("Số điện thoại phải có 10 chữ số, bắt đầu bằng 0.")
    if not re.fullmatch(r"(?:[0-9A-F]{2}:)+[0-9A-F]{2}", uid):
        return json_error("UID RFID không đúng định dạng.")

    cleanup_pending_faces()
    face = PENDING_FACES.get(face_token)
    if (
        not face
        or face["user_id"] != session["user_id"]
        or time.time() - face["created_at"] > 300
    ):
        return json_error(
            "Khuôn mặt chưa được xử lý hoặc token đã hết hạn. Hãy chụp lại."
        )

    encoding = face.get("encoding")
    if (
        not isinstance(encoding, list)
        or len(encoding) != 128
        or not np.isfinite(encoding).all()
    ):
        return json_error("Dữ liệu khuôn mặt không hợp lệ.")

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")

        # Kiểm tra UID đã được đăng ký chưa.
        uid_exists = conn.execute(
            """
            SELECT driver_id
            FROM rfid_cards
            WHERE UPPER(uid) = ?
            LIMIT 1
            """,
            (uid,),
        ).fetchone()

        if uid_exists:
            conn.rollback()
            return json_error(
                f"UID RFID đã được đăng ký cho tài xế "
                f"{uid_exists['driver_id']}. Vui lòng dùng thẻ khác.",
                409,
            )

        # Chặn đăng ký lại khuôn mặt đã tồn tại.
        existing_faces = conn.execute(
            "SELECT driver_id, encoding_json FROM face_encodings"
        ).fetchall()

        new_encoding = np.asarray(encoding, dtype=np.float64)

        for existing_driver_id, encoding_json in existing_faces:
            try:
                known_encoding = np.asarray(
                    json.loads(encoding_json),
                    dtype=np.float64,
                )

                if (
                    known_encoding.shape != (128,)
                    or not np.isfinite(known_encoding).all()
                ):
                    continue

                distance = float(
                    np.linalg.norm(new_encoding - known_encoding)
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            if distance < 0.42:
                conn.rollback()
                return json_error(
                    f"Khuôn mặt có thể đã đăng ký cho tài xế "
                    f"{existing_driver_id}. Vui lòng kiểm tra hồ sơ hiện có.",
                    409,
                )

        conn.execute(
            """
            INSERT INTO drivers (driver_id, full_name, phone, status)
            VALUES (?, ?, ?, 'ACTIVE')
            """,
            (driver_id, full_name, phone),
        )
        conn.execute(
            "INSERT INTO rfid_cards (uid, driver_id) VALUES (?, ?)",
            (uid, driver_id),
        )
        conn.execute(
            """
            INSERT INTO face_encodings
                (driver_id, encoding_json, samples, updated_at)
            VALUES (?, ?, 1, datetime('now', 'localtime'))
            """,
            (driver_id, json.dumps(encoding)),
        )

        conn.commit()
        PENDING_FACES.pop(face_token, None)
        return jsonify({
            "success": True,
            "message": "Đăng ký tài xế thành công.",
        }), 201

    except sqlite3.IntegrityError:
        conn.rollback()
        return json_error("Mã tài xế, UID hoặc hồ sơ khuôn mặt đã tồn tại.", 409)
    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi lưu hồ sơ tài xế")
        return json_error("Lỗi cơ sở dữ liệu. Hồ sơ chưa được lưu.", 500)
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
        return json_error("Số điện thoại phải có 10 chữ số, bắt đầu bằng 0.")
    if status_value not in ("ACTIVE", "INACTIVE"):
        return json_error("Trạng thái chỉ được là ACTIVE hoặc INACTIVE.")

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """
            UPDATE drivers
            SET full_name = ?, phone = ?, status = ?
            WHERE driver_id = ?
            """,
            (full_name, phone, status_value, driver_id),
        )

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
        exists = conn.execute(
            "SELECT 1 FROM drivers WHERE driver_id = ?",
            (driver_id,),
        ).fetchone()

        if exists is None:
            conn.rollback()
            return json_error("Không tìm thấy tài xế.", 404)

        # Chỉ xóa hồ sơ hiện hành; không xóa parking_history.
        conn.execute(
            "DELETE FROM face_encodings WHERE driver_id = ?",
            (driver_id,),
        )
        conn.execute(
            "DELETE FROM rfid_cards WHERE driver_id = ?",
            (driver_id,),
        )
        conn.execute(
            "DELETE FROM drivers WHERE driver_id = ?",
            (driver_id,),
        )

        conn.commit()
        return jsonify({
            "success": True,
            "message": "Đã xóa hồ sơ tài xế. Lịch sử bãi xe được giữ nguyên.",
        })
    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi xóa tài xế")
        return json_error("Không thể xóa tài xế.", 500)
    finally:
        conn.close()
