import base64
import json
import sqlite3
import time
import re

import cv2
import face_recognition
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from services.app_context import get_db, json_error
from services.auth_utils import login_required


face_bp = Blueprint("face_api", __name__)


@face_bp.route("/api/face/pending")
@login_required
def face_auth_pending():
    conn = get_db()
    try:
        now = time.time()

        # Hết hạn yêu cầu chờ quá 35 giây.
        conn.execute("""
            UPDATE face_auth_requests
            SET state = 'EXPIRED',
                processed_at = ?,
                message = 'Hết thời gian xác thực'
            WHERE state = 'PENDING' AND created_at < ?
        """, (now, now - 35))
        conn.commit()

        # Trạng thái mới nhất để giao diện hiện cả lỗi RFID.
        latest = conn.execute("""
            SELECT id, uid, driver_id, state, created_at,
                   processed_at, message
            FROM face_auth_requests
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        latest_status = None
        if latest:
            latest_status = {
                "id": latest["id"],
                "uid": latest["uid"],
                "driver_id": latest["driver_id"],
                "state": latest["state"],
                "created_at": latest["created_at"],
                "processed_at": latest["processed_at"],
                "message": latest["message"],
            }

        # Giữ nguyên cơ chế lấy yêu cầu đang chờ.
        row = conn.execute("""
            SELECT q.id, q.uid, q.driver_id, d.full_name
            FROM face_auth_requests q
            LEFT JOIN drivers d ON d.driver_id = q.driver_id
            WHERE q.state = 'PENDING'
            ORDER BY q.id ASC
            LIMIT 1
        """).fetchone()

        if not row:
            return jsonify({
                "success": True,
                "pending": False,
                "latest_status": latest_status,
            })

        return jsonify({
            "success": True,
            "pending": True,
            "request_id": row["id"],
            "driver_id": row["driver_id"],
            "full_name": row["full_name"],
            "uid": row["uid"],
            "latest_status": latest_status,
        })
    finally:
        conn.close()


@face_bp.route("/api/face/verify", methods=["POST"])
@login_required
def verify_driver_face():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("Dữ liệu không hợp lệ.")

    try:
        request_id = int(data.get("request_id"))
    except (TypeError, ValueError):
        return json_error("Mã yêu cầu xác thực không hợp lệ.")

    image_data = data.get("image", "")
    if not isinstance(image_data, str):
        return json_error("Ảnh không hợp lệ.")

    match = re.fullmatch(
        r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)",
        image_data,
    )
    if not match:
        return json_error("Ảnh phải ở định dạng JPEG.")

    conn = get_db()
    try:
        # Bước 1: Đọc yêu cầu, không giữ khóa ghi khi xử lý ảnh.
        req = conn.execute("""
            SELECT id, uid, driver_id, state, created_at
            FROM face_auth_requests
            WHERE id = ?
        """, (request_id,)).fetchone()

        if not req or req["state"] != "PENDING":
            return json_error("Yêu cầu không còn hiệu lực.", 409)

        if time.time() - req["created_at"] > 35:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'EXPIRED',
                    processed_at = ?,
                    message = 'Hết thời gian xác thực'
                WHERE id = ? AND state = 'PENDING'
            """, (time.time(), request_id))
            conn.commit()
            return json_error("Yêu cầu đã hết hạn.", 409)

        # Bước 2: RFID phải thuộc đúng tài xế ACTIVE có dữ liệu khuôn mặt.
        driver = conn.execute("""
            SELECT d.driver_id, f.encoding_json
            FROM rfid_cards AS r
            JOIN drivers AS d ON d.driver_id = r.driver_id
            JOIN face_encodings AS f ON f.driver_id = d.driver_id
            WHERE UPPER(r.uid) = UPPER(?)
              AND d.driver_id = ?
              AND d.status = 'ACTIVE'
            LIMIT 1
        """, (req["uid"], req["driver_id"])).fetchone()

        if not driver:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'DENY',
                    processed_at = ?,
                    message = 'Tài xế không hợp lệ hoặc chưa đăng ký khuôn mặt'
                WHERE id = ? AND state = 'PENDING'
            """, (time.time(), request_id))
            conn.commit()
            return jsonify({
                "success": True,
                "verified": False,
                "message": "Tài xế không hợp lệ hoặc chưa đăng ký khuôn mặt.",
            })

        original_encoding_json = driver["encoding_json"]

        # Bước 3: Giải mã và nhận diện ảnh ngoài giao dịch ghi SQLite.
        try:
            raw = base64.b64decode(match.group(1), validate=True)
            if not raw or len(raw) > 5 * 1024 * 1024:
                raise ValueError("Ảnh rỗng hoặc vượt quá 5 MB.")

            array = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Không đọc được ảnh JPEG.")

            height, width = bgr.shape[:2]
            if height < 1 or width < 1:
                raise ValueError("Kích thước ảnh không hợp lệ.")

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
                raise ValueError(
                    f"Ảnh phải có đúng một khuôn mặt; "
                    f"phát hiện {len(locations)}."
                )

            encodings = face_recognition.face_encodings(
                rgb,
                known_face_locations=locations,
            )
            if len(encodings) != 1 or not np.isfinite(encodings[0]).all():
                raise ValueError("Không trích xuất được khuôn mặt.")

            known = np.asarray(
                json.loads(original_encoding_json),
                dtype=np.float64,
            )
            if known.shape != (128,) or not np.isfinite(known).all():
                raise ValueError("Dữ liệu khuôn mặt đã lưu không hợp lệ.")

            distance = float(np.linalg.norm(encodings[0] - known))

        except (ValueError, TypeError, json.JSONDecodeError, cv2.error) as exc:
            # Ảnh lỗi được chụp lại; không tự DENY yêu cầu còn hạn.
            return json_error(
                f"Không xác thực được ảnh: {str(exc)[:180]} Hãy chụp lại.",
                400,
            )

        # Bước 4: Xác thực xong mới mở giao dịch ghi ngắn.
        verified = distance < 0.45
        state = "ALLOW" if verified else "DENY"
        message = "Xác thực thành công" if verified else "Khuôn mặt không khớp"

        conn.execute("BEGIN IMMEDIATE")

        # Kiểm tra lại yêu cầu vì trạng thái có thể đổi trong lúc xử lý ảnh.
        current = conn.execute("""
            SELECT state, created_at
            FROM face_auth_requests
            WHERE id = ?
        """, (request_id,)).fetchone()

        if not current or current["state"] != "PENDING":
            conn.rollback()
            return json_error(
                "Yêu cầu đã được xử lý hoặc không còn hiệu lực.",
                409,
            )

        if time.time() - current["created_at"] > 35:
            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'EXPIRED',
                    processed_at = ?,
                    message = 'Hết thời gian xác thực'
                WHERE id = ? AND state = 'PENDING'
            """, (time.time(), request_id))
            conn.commit()
            return json_error("Yêu cầu đã hết hạn.", 409)

        # Xác nhận thẻ, tài xế và dữ liệu khuôn mặt vẫn như lúc kiểm tra.
        current_driver = conn.execute("""
            SELECT d.driver_id, f.encoding_json
            FROM rfid_cards AS r
            JOIN drivers AS d ON d.driver_id = r.driver_id
            JOIN face_encodings AS f ON f.driver_id = d.driver_id
            WHERE UPPER(r.uid) = UPPER(?)
              AND d.driver_id = ?
              AND d.status = 'ACTIVE'
            LIMIT 1
        """, (req["uid"], req["driver_id"])).fetchone()

        if (
            not current_driver
            or current_driver["encoding_json"] != original_encoding_json
        ):
            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'DENY',
                    processed_at = ?,
                    message = 'Hồ sơ tài xế đã thay đổi'
                WHERE id = ? AND state = 'PENDING'
            """, (time.time(), request_id))
            conn.commit()
            return jsonify({
                "success": True,
                "verified": False,
                "message": "Hồ sơ tài xế đã thay đổi. Hãy quẹt thẻ lại.",
            })

        cursor = conn.execute("""
            UPDATE face_auth_requests
            SET state = ?,
                processed_at = ?,
                message = ?
            WHERE id = ? AND state = 'PENDING'
        """, (state, time.time(), message, request_id))

        if cursor.rowcount != 1:
            conn.rollback()
            return json_error("Không thể xác nhận yêu cầu.", 409)

        conn.commit()
        return jsonify({
            "success": True,
            "verified": verified,
            "message": message,
        })

    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception("Lỗi xác thực khuôn mặt")
        return json_error("Lỗi cơ sở dữ liệu khi xác thực.", 500)
    finally:
        conn.close()
