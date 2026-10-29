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
    gate = request.args.get("gate", "ENTRY").strip().upper()

    if gate not in ("ENTRY", "EXIT"):
        return json_error("Cổng xác thực không hợp lệ.", 400)

    conn = get_db()
    try:
        now = time.time()

        # Hết hạn yêu cầu PENDING theo đúng cổng.
        conn.execute("""
            UPDATE face_auth_requests
            SET state = 'EXPIRED',
                processed_at = ?,
                message = 'Hết thời gian xác thực'
            WHERE state = 'PENDING'
              AND gate_type = ?
              AND created_at < ?
        """, (now, gate, now - 35))
        conn.commit()

        # Trạng thái gần nhất của đúng cổng.
        latest = conn.execute("""
            SELECT id, uid, driver_id, state, created_at,
                   processed_at, message
            FROM face_auth_requests
            WHERE gate_type = ?
            ORDER BY id DESC
            LIMIT 1
        """, (gate,)).fetchone()

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

        # Chỉ lấy yêu cầu đang chờ của đúng cổng.
        row = conn.execute("""
            SELECT q.id, q.uid, q.driver_id, d.full_name
            FROM face_auth_requests q
            LEFT JOIN drivers d ON d.driver_id = q.driver_id
            WHERE q.state = 'PENDING'
              AND q.gate_type = ?
            ORDER BY q.id ASC
            LIMIT 1
        """, (gate,)).fetchone()

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
            "gate": gate,
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

    gate = str(data.get("gate", "")).strip().upper()
    if gate not in ("ENTRY", "EXIT"):
        return json_error("Cổng xác thực không hợp lệ.", 400)

    images = data.get("images")

    if not isinstance(images, list) or len(images) != 5:
        return json_error(
            "Cần gửi đúng 5 khung hình để xác thực.",
            400,
        )    

    conn = get_db()
    try:
        # Bước 1: Đọc yêu cầu, không giữ khóa ghi khi xử lý ảnh.
        req = conn.execute("""
            SELECT id, uid, driver_id, state, created_at, gate_type
            FROM face_auth_requests
            WHERE id = ?
        """, (request_id,)).fetchone()

        if not req or req["state"] != "PENDING":
            return json_error("Yêu cầu không còn hiệu lực.", 409)

        if req["gate_type"] != gate:
            return json_error(
                "Yêu cầu thuộc cổng khác, không thể xác thực.", 403
            )

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

        # Bước 3: Xử lý nhiều khung hình ngoài giao dịch ghi SQLite.
        try:
            known = np.asarray(
                json.loads(original_encoding_json),
                dtype=np.float64,
            )

            if known.shape != (128,) or not np.isfinite(known).all():
                raise ValueError("Dữ liệu khuôn mặt đã lưu không hợp lệ.")

            query_encodings = []

            for image_data in images:
                if not isinstance(image_data, str):
                    continue

                match = re.fullmatch(
                    r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)",
                    image_data,
                )
                if not match:
                    continue

                try:
                    raw = base64.b64decode(match.group(1), validate=True)

                    if not raw or len(raw) > 5 * 1024 * 1024:
                        continue

                    array = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

                    if bgr is None:
                        continue

                    height, width = bgr.shape[:2]
                    if height < 1 or width < 1:
                        continue

                    if max(height, width) > 960:
                        scale = 960 / max(height, width)
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
                        continue

                    found = face_recognition.face_encodings(
                        rgb,
                        known_face_locations=locations,
                    )

                    if len(found) != 1:
                        continue

                    encoding = np.asarray(found[0], dtype=np.float64)

                    if (
                        encoding.shape == (128,)
                        and np.isfinite(encoding).all()
                    ):
                        query_encodings.append(encoding)

                except (ValueError, TypeError, cv2.error):
                    continue

            if len(query_encodings) < 3:
                return json_error(
                    f"Chỉ nhận diện được {len(query_encodings)}/5 khung hình. "
                    "Hãy nhìn thẳng camera, đủ sáng và chụp lại.",
                    400,
                )

        except (ValueError, TypeError, json.JSONDecodeError, cv2.error) as exc:
            return json_error(
                f"Không xử lý được ảnh: {str(exc)[:150]} Hãy chụp lại.",
                400,
            )            

        # Bước 4: Xác thực xong mới mở giao dịch ghi ngắn.
        # So sánh từng khung hình với tài xế mục tiêu và các tài xế khác.
        other_faces = conn.execute(
            """
            SELECT driver_id, encoding_json
            FROM face_encodings
            WHERE driver_id != ?
            """,
            (req["driver_id"],),
        ).fetchall()

        other_encodings = []

        for other in other_faces:
            try:
                other_encoding = np.asarray(
                    json.loads(other["encoding_json"]),
                    dtype=np.float64,
                )

                if (
                    other_encoding.shape == (128,)
                    and np.isfinite(other_encoding).all()
                ):
                    other_encodings.append(
                        (other["driver_id"], other_encoding)
                    )

            except (TypeError, ValueError, json.JSONDecodeError):
                continue

        threshold = 0.45
        margin = 0.05

        frame_distances = []
        matching_frames = 0

        for query in query_encodings:
            distance = float(np.linalg.norm(query - known))
            frame_distances.append(distance)

            if not other_encodings:
                continue

            closest_other_distance = min(
                float(np.linalg.norm(query - other_encoding))
                for _, other_encoding in other_encodings
            )

            if (
                distance < threshold
                and distance + margin < closest_other_distance
            ):
                matching_frames += 1

        required_matches = max(
            3,
            int(np.ceil(0.6 * len(query_encodings))),
        )

        verified = (
            bool(other_encodings)
            and matching_frames >= required_matches
        )

        median_distance = float(np.median(frame_distances))

        current_app.logger.warning(
            "[FACE VERIFY MULTI] request_id=%s uid=%s driver_id=%s "
            "valid_frames=%d matches=%d required=%d median_distance=%.6f "
            "threshold=%.2f margin=%.2f verified=%s",
            request_id,
            req["uid"],
            req["driver_id"],
            len(query_encodings),
            matching_frames,
            required_matches,
            median_distance,
            threshold,
            margin,
            verified,
        )


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
            # ////////////////////////////////////////////////////////////

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
