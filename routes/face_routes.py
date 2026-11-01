
import base64
import binascii
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

FACE_REQUEST_TTL = 120
FACE_MATCH_THRESHOLD = 0.45
FACE_MATCH_MARGIN = 0.05
MIN_VALID_QUERY_FRAMES = 3
VERIFY_FRAME_COUNT = 5


def load_valid_embedding(encoding_json):
    """Đọc và kiểm tra embedding 128 chiều."""
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


def decode_query_image(image_data):
    """Giải mã ảnh JPEG và lấy embedding khuôn mặt."""
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

    except (
        ValueError,
        TypeError,
        binascii.Error,
        cv2.error,
    ):
        return None


def load_face_profiles(conn):
    """
    Đọc hồ sơ khuôn mặt của các tài xế ACTIVE.

    Ưu tiên các mẫu riêng lẻ trong face_embedding_samples.
    Chỉ dùng face_encodings làm dự phòng khi tài xế chưa có
    bất kỳ bản ghi mẫu riêng lẻ nào.
    """

    rows = conn.execute("""
        SELECT
            d.driver_id,
            f.encoding_json
        FROM drivers AS d
        JOIN face_encodings AS f
            ON f.driver_id = d.driver_id
        WHERE d.status = 'ACTIVE'
        ORDER BY d.driver_id
    """).fetchall()

    profiles = {}
    signatures = {}

    for row in rows:
        driver_id = row["driver_id"]

        sample_rows = conn.execute("""
            SELECT id, pose, encoding_json
            FROM face_embedding_samples
            WHERE driver_id = ?
            ORDER BY id
        """, (driver_id,)).fetchall()

        embeddings = []
        signature_items = []

        if sample_rows:
            # Có mẫu riêng lẻ thì không âm thầm quay về
            # vector cũ nếu các mẫu này bị lỗi.
            for sample in sample_rows:
                signature_items.append((
                    sample["id"],
                    sample["pose"],
                    sample["encoding_json"],
                ))

                vector = load_valid_embedding(
                    sample["encoding_json"]
                )

                if vector is not None:
                    embeddings.append(vector)

        else:
            # Tương thích hồ sơ cũ chưa có bảng mẫu riêng.
            legacy_json = row["encoding_json"]

            signature_items.append((
                None,
                "LEGACY",
                legacy_json,
            ))

            vector = load_valid_embedding(legacy_json)

            if vector is not None:
                embeddings.append(vector)

        signatures[driver_id] = tuple(signature_items)

        # Hồ sơ đã đăng ký nhưng không còn embedding hợp lệ:
        # không được bỏ qua và tiếp tục xác thực.
        if not embeddings:
            raise ValueError(
                f"Hồ sơ khuôn mặt của tài xế {driver_id} "
                "không có embedding hợp lệ."
            )

        profiles[driver_id] = np.stack(embeddings)

    return profiles, signatures




def profile_distance(query_encoding, known_encodings):
    """
    Tính khoảng cách ổn định với một hồ sơ.

    Lấy trung vị của tối đa 3 mẫu gần nhất.
    Không dùng duy nhất khoảng cách nhỏ nhất trên toàn bộ mẫu.
    """

    distances = np.linalg.norm(
        known_encodings - query_encoding,
        axis=1,
    )

    closest = np.sort(distances)[:min(3, len(distances))]

    return float(np.median(closest))

def expire_request(conn, request_id):
    """Chuyển yêu cầu hết hạn nếu vẫn đang PENDING."""
    now = time.time()

    conn.execute("""
        UPDATE face_auth_requests
        SET state = 'EXPIRED',
            processed_at = ?,
            message = 'Hết thời gian xác thực'
        WHERE id = ?
          AND state = 'PENDING'
    """, (now, request_id))

def get_global_pending_request(conn):
    """
    Lấy yêu cầu PENDING đầu tiên trên toàn hệ thống,
    không phân biệt ENTRY hay EXIT.
    """
    return conn.execute("""
        SELECT id, gate_type
        FROM face_auth_requests
        WHERE state = 'PENDING'
        ORDER BY id ASC
        LIMIT 1
    """).fetchone()

def face_auth_pending():
    gate = request.args.get("gate", "ENTRY").strip().upper()

    if gate not in ("ENTRY", "EXIT"):
        return json_error("Cổng xác thực không hợp lệ.", 400)

    conn = get_db()

    try:
        now = time.time()

        # Hết hạn các yêu cầu PENDING quá thời gian trên toàn hệ thống.
        conn.execute("""
            UPDATE face_auth_requests
            SET state = 'EXPIRED',
                processed_at = ?,
                message = 'Hết thời gian xác thực'
            WHERE state = 'PENDING'
              AND created_at < ?
        """, (now, now - FACE_REQUEST_TTL))

        conn.commit()

        # Trạng thái gần nhất của riêng cổng đang hỏi.
        latest = conn.execute("""
            SELECT
                id,
                uid,
                driver_id,
                state,
                created_at,
                processed_at,
                message
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

        # Chỉ lấy một yêu cầu PENDING đầu tiên toàn hệ thống.
        head = get_global_pending_request(conn)

        # Không còn yêu cầu nào đang chờ.
        if not head:
            return jsonify({
                "success": True,
                "pending": False,
                "waiting": False,
                "latest_status": latest_status,
            })

        # Đầu hàng thuộc cổng còn lại: cổng hiện tại phải chờ.
        if head["gate_type"] != gate:
            return jsonify({
                "success": True,
                "pending": False,
                "waiting": True,
                "queue_head_gate": head["gate_type"],
                "queue_head_request_id": head["id"],
                "latest_status": latest_status,
            })

        # Đầu hàng thuộc đúng cổng đang hỏi.
        row = conn.execute("""
            SELECT
                q.id,
                q.uid,
                q.driver_id,
                d.full_name
            FROM face_auth_requests AS q
            LEFT JOIN drivers AS d
                ON d.driver_id = q.driver_id
            WHERE q.id = ?
              AND q.state = 'PENDING'
            LIMIT 1
        """, (head["id"],)).fetchone()

        if not row:
            return jsonify({
                "success": True,
                "pending": False,
                "waiting": False,
                "latest_status": latest_status,
            })

        return jsonify({
            "success": True,
            "pending": True,
            "waiting": False,
            "request_id": row["id"],
            "driver_id": row["driver_id"],
            "full_name": row["full_name"],
            "uid": row["uid"],
            "gate": gate,
            "latest_status": latest_status,
        })

    except sqlite3.Error:
        conn.rollback()

        current_app.logger.exception(
            "Lỗi lấy yêu cầu xác thực khuôn mặt"
        )

        return json_error(
            "Không thể lấy yêu cầu xác thực.",
            500,
        )

    finally:
        conn.close()

face_bp.add_url_rule(
    "/api/face/pending",
    view_func=login_required(face_auth_pending),
    methods=["GET"],
)


@face_bp.route("/api/face/verify", methods=["POST"])
@login_required
def verify_driver_face():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return json_error("Dữ liệu không hợp lệ.")

    try:
        request_id = int(data.get("request_id"))
    except (TypeError, ValueError):
        return json_error(
            "Mã yêu cầu xác thực không hợp lệ."
        )

    gate = str(data.get("gate", "")).strip().upper()

    if gate not in ("ENTRY", "EXIT"):
        return json_error(
            "Cổng xác thực không hợp lệ.",
            400,
        )

    images = data.get("images")

    if (
        not isinstance(images, list)
        or len(images) != VERIFY_FRAME_COUNT
    ):
        return json_error(
            f"Cần gửi đúng {VERIFY_FRAME_COUNT} khung hình để xác thực.",
            400,
        )

    conn = get_db()
    try:
        # --------------------------------------------------
        # BƯỚC 1: Hết hạn yêu cầu cũ và kiểm tra yêu cầu.
        # --------------------------------------------------

        now = time.time()

        conn.execute("""
            UPDATE face_auth_requests
            SET state = 'EXPIRED',
                processed_at = ?,
                message = 'Hết thời gian xác thực'
            WHERE state = 'PENDING'
              AND created_at < ?
        """, (now, now - FACE_REQUEST_TTL))

        conn.commit()    

        req = conn.execute("""
            SELECT
                id,
                uid,
                driver_id,
                state,
                created_at,
                gate_type
            FROM face_auth_requests
            WHERE id = ?
        """, (request_id,)).fetchone()

        if not req or req["state"] != "PENDING":
            return json_error(
                "Yêu cầu không còn hiệu lực.",
                409,
            )

        if req["gate_type"] != gate:
            return json_error(
                "Yêu cầu thuộc cổng khác, không thể xác thực.",
                403,
            )

        # Yêu cầu phải là đầu hàng PENDING chung ENTRY + EXIT.
        queue_head = get_global_pending_request(conn)

        if (
            queue_head is None
            or int(queue_head["id"]) != request_id
        ):
            return json_error(
                "Yêu cầu chưa đến lượt xác thực trong FIFO chung. "
                "Vui lòng chờ cổng đang được xử lý.",
                409,
            )            

        # --------------------------------------------------
        # BƯỚC 1.1: BẮT BUỘC BIỂN SỐ ĐÃ ĐƯỢC XÁC THỰC
        # --------------------------------------------------

        plate_auth = conn.execute("""
            SELECT
                uid,
                driver_id,
                gate_type,
                status,
                detected_plate,
                registered_plate
            FROM plate_auth_results
            WHERE request_id = ?
            LIMIT 1
        """, (request_id,)).fetchone()

        if not plate_auth:
            return json_error(
                "Chưa xác thực biển số xe. "
                "Vui lòng thực hiện xác thực biển số trước.",
                403,
            )

        if (
            plate_auth["status"] != "ALLOW"
            or plate_auth["uid"] != req["uid"]
            or plate_auth["driver_id"] != req["driver_id"]
            or plate_auth["gate_type"] != gate
            or not plate_auth["detected_plate"]
            or not plate_auth["registered_plate"]
            or plate_auth["detected_plate"] != plate_auth["registered_plate"]
        ):
            return json_error(
                "Biển số xe không hợp lệ hoặc không khớp với yêu cầu RFID.",
                403,
            )

        if time.time() - float(req["created_at"]) > FACE_REQUEST_TTL:
            conn.execute("BEGIN IMMEDIATE")
            expire_request(conn, request_id)
            conn.commit()

            return json_error(
                "Yêu cầu đã hết hạn.",
                409,
            )

        # --------------------------------------------------
        # BƯỚC 2: Kiểm tra RFID và tài xế.
        # --------------------------------------------------

        driver = conn.execute("""
            SELECT
                d.driver_id
            FROM rfid_cards AS r
            JOIN drivers AS d
                ON d.driver_id = r.driver_id
            JOIN face_encodings AS f
                ON f.driver_id = d.driver_id
            WHERE UPPER(r.uid) = UPPER(?)
              AND d.driver_id = ?
              AND d.status = 'ACTIVE'
            LIMIT 1
        """, (
            req["uid"],
            req["driver_id"],
        )).fetchone()

        if not driver:
            conn.execute("BEGIN IMMEDIATE")

            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'DENY',
                    processed_at = ?,
                    message = 'Tài xế không hợp lệ hoặc chưa đăng ký khuôn mặt'
                WHERE id = ?
                  AND state = 'PENDING'
            """, (time.time(), request_id))

            conn.commit()

            return jsonify({
                "success": True,
                "verified": False,
                "message": (
                    "Tài xế không hợp lệ hoặc chưa đăng ký khuôn mặt."
                ),
            })

        target_driver_id = driver["driver_id"]

        # Đọc hồ sơ và lưu dấu vết dữ liệu trước khi xử lý ảnh.
        profiles, original_signatures = load_face_profiles(conn)

        if target_driver_id not in profiles:
            conn.execute("BEGIN IMMEDIATE")

            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'DENY',
                    processed_at = ?,
                    message = 'Không có mẫu khuôn mặt hợp lệ'
                WHERE id = ?
                  AND state = 'PENDING'
            """, (time.time(), request_id))

            conn.commit()

            return jsonify({
                "success": True,
                "verified": False,
                "message": "Không tìm thấy mẫu khuôn mặt hợp lệ.",
            })

        target_encodings = profiles[target_driver_id]

        other_profiles = {
            driver_id: encodings
            for driver_id, encodings in profiles.items()
            if driver_id != target_driver_id
        }

        # --------------------------------------------------
        # BƯỚC 3: Xử lý 5 ảnh, không khóa ghi SQLite.
        # --------------------------------------------------

        query_encodings = []

        for image_data in images:
            encoding = decode_query_image(image_data)

            if encoding is not None:
                query_encodings.append(encoding)

        if len(query_encodings) < MIN_VALID_QUERY_FRAMES:
            return json_error(
                f"Chỉ nhận diện được {len(query_encodings)}/5 "
                "khung hình. Hãy nhìn camera, đủ sáng và chụp lại.",
                400,
            )

        # --------------------------------------------------
        # BƯỚC 4: So sánh nhiều mẫu khuôn mặt.
        # --------------------------------------------------

        matching_frames = 0
        target_distances = []
        competitor_distances = []

        for query in query_encodings:
            target_distance = profile_distance(
                query,
                target_encodings,
            )

            target_distances.append(target_distance)

            # Nếu chỉ có một tài xế hợp lệ, chỉ cần đạt
            # ngưỡng khuôn mặt. Không tự động DENY vì thiếu đối thủ.
            if not other_profiles:
                if target_distance < FACE_MATCH_THRESHOLD:
                    matching_frames += 1
                continue

            distances_to_others = {
                other_id: profile_distance(
                    query,
                    other_encodings,
                )
                for other_id, other_encodings in other_profiles.items()
            }

            closest_other_distance = min(
                distances_to_others.values()
            )

            competitor_distances.append(closest_other_distance)

            if (
                target_distance < FACE_MATCH_THRESHOLD
                and target_distance + FACE_MATCH_MARGIN
                < closest_other_distance
            ):
                matching_frames += 1

        required_matches = max(
            3,
            int(np.ceil(0.6 * len(query_encodings))),
        )

        verified = matching_frames >= required_matches

        median_target_distance = float(
            np.median(target_distances)
        )

        median_competitor_distance = (
            float(np.median(competitor_distances))
            if competitor_distances
            else None
        )

        current_app.logger.info(
            "[FACE VERIFY MULTI] "
            "request_id=%s gate=%s uid=%s driver_id=%s "
            "valid_frames=%d target_samples=%d "
            "other_drivers=%d matches=%d required=%d "
            "target_median=%.6f competitor_median=%s "
            "threshold=%.2f margin=%.2f verified=%s",
            request_id,
            gate,
            req["uid"],
            target_driver_id,
            len(query_encodings),
            len(target_encodings),
            len(other_profiles),
            matching_frames,
            required_matches,
            median_target_distance,
            (
                f"{median_competitor_distance:.6f}"
                if median_competitor_distance is not None
                else "NONE"
            ),
            FACE_MATCH_THRESHOLD,
            FACE_MATCH_MARGIN,
            verified,
        )

        state = "ALLOW" if verified else "DENY"

        message = (
            "Xác thực thành công"
            if verified
            else "Khuôn mặt không khớp"
        )

        # --------------------------------------------------
        # BƯỚC 5: Giao dịch ghi ngắn và kiểm tra lại dữ liệu.
        # --------------------------------------------------

        conn.execute("BEGIN IMMEDIATE")

        current = conn.execute("""
            SELECT state, created_at, gate_type
            FROM face_auth_requests
            WHERE id = ?
        """, (request_id,)).fetchone()

        if (
            not current
            or current["state"] != "PENDING"
            or current["gate_type"] != gate
        ):
            conn.rollback()

            return json_error(
                "Yêu cầu đã được xử lý hoặc không còn hiệu lực.",
                409,
            )

        # Kiểm tra lại đầu hàng bên trong giao dịch ghi.
        # BEGIN IMMEDIATE đang giữ quyền ghi SQLite,
        # nên yêu cầu khác không thể chen vào giữa bước kiểm tra và cập nhật.
        queue_head = get_global_pending_request(conn)

        if (
            queue_head is None
            or int(queue_head["id"]) != request_id
        ):
            conn.rollback()

            return json_error(
                "Thứ tự FIFO đã thay đổi. "
                "Yêu cầu này không còn đứng đầu hàng.",
                409,
            )    

        # Kiểm tra lại kết quả biển số trong giao dịch ghi.
        current_plate = conn.execute("""
            SELECT
                uid,
                driver_id,
                gate_type,
                status,
                detected_plate,
                registered_plate
            FROM plate_auth_results
            WHERE request_id = ?
            LIMIT 1
        """, (request_id,)).fetchone()

        if (
            not current_plate
            or current_plate["status"] != "ALLOW"
            or current_plate["uid"] != req["uid"]
            or current_plate["driver_id"] != req["driver_id"]
            or current_plate["gate_type"] != gate
            or not current_plate["detected_plate"]
            or not current_plate["registered_plate"]
            or current_plate["detected_plate"]
                != current_plate["registered_plate"]
        ):
            conn.rollback()

            return json_error(
                "Không thể xác thực khuôn mặt: "
                "biển số chưa được xác thực hợp lệ.",
                403,
            )                   

        if time.time() - float(current["created_at"]) > FACE_REQUEST_TTL:
            expire_request(conn, request_id)
            conn.commit()

            return json_error(
                "Yêu cầu đã hết hạn.",
                409,
            )

        # Kiểm tra lại RFID và trạng thái tài xế.
        current_driver = conn.execute("""
            SELECT d.driver_id
            FROM rfid_cards AS r
            JOIN drivers AS d
                ON d.driver_id = r.driver_id
            JOIN face_encodings AS f
                ON f.driver_id = d.driver_id
            WHERE UPPER(r.uid) = UPPER(?)
              AND d.driver_id = ?
              AND d.status = 'ACTIVE'
            LIMIT 1
        """, (
            req["uid"],
            target_driver_id,
        )).fetchone()

        if not current_driver:
            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'DENY',
                    processed_at = ?,
                    message = 'Hồ sơ tài xế hoặc RFID đã thay đổi'
                WHERE id = ?
                  AND state = 'PENDING'
            """, (time.time(), request_id))

            conn.commit()

            return jsonify({
                "success": True,
                "verified": False,
                "message": (
                    "Hồ sơ tài xế hoặc RFID đã thay đổi. "
                    "Hãy quẹt thẻ lại."
                ),
            })

        # Kiểm tra lại toàn bộ mẫu khuôn mặt.
        _, current_signatures = load_face_profiles(conn)

        if current_signatures != original_signatures:
            conn.execute("""
                UPDATE face_auth_requests
                SET state = 'DENY',
                    processed_at = ?,
                    message = 'Mẫu khuôn mặt đã thay đổi trong lúc xác thực'
                WHERE id = ?
                  AND state = 'PENDING'
            """, (time.time(), request_id))

            conn.commit()

            return jsonify({
                "success": True,
                "verified": False,
                "message": (
                    "Mẫu khuôn mặt đã thay đổi. "
                    "Vui lòng quẹt thẻ và xác thực lại."
                ),
            })

        cursor = conn.execute("""
            UPDATE face_auth_requests
            SET state = ?,
                processed_at = ?,
                message = ?
            WHERE id = ?
              AND state = 'PENDING'
        """, (
            state,
            time.time(),
            message,
            request_id,
        ))

        if cursor.rowcount != 1:
            conn.rollback()

            return json_error(
                "Không thể xác nhận yêu cầu.",
                409,
            )

        conn.commit()

        return jsonify({
            "success": True,
            "verified": verified,
            "message": message,
        })

    except sqlite3.Error:
        conn.rollback()

        current_app.logger.exception(
            "Lỗi cơ sở dữ liệu khi xác thực khuôn mặt"
        )

        return json_error(
            "Lỗi cơ sở dữ liệu khi xác thực.",
            500,
        )

    except (TypeError, ValueError, KeyError):
        conn.rollback()

        current_app.logger.exception(
            "Lỗi dữ liệu trong quá trình xác thực khuôn mặt"
        )

        return json_error(
            "Dữ liệu xác thực không hợp lệ.",
            400,
        )

    finally:
        conn.close()
