import sqlite3
import time

from flask import Blueprint, current_app, jsonify, request
from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    request,
    stream_with_context,
)

from services.app_context import get_db, json_error
from services.auth_utils import login_required
from services.plate_camera_service import (
    camera_manager,
    capture_plate_burst,
    normalize_plate_text,
)



plate_camera_bp = Blueprint("plate_camera", __name__)

FACE_REQUEST_TTL = 120


def get_global_pending_request(conn):
    return conn.execute("""
        SELECT id, gate_type
        FROM face_auth_requests
        WHERE state = 'PENDING'
        ORDER BY id ASC
        LIMIT 1
    """).fetchone()


def get_registered_plates(conn, driver_id, uid):
    rows = conn.execute("""
        SELECT DISTINCT v.plate_number
        FROM vehicles AS v
        JOIN drivers AS d
            ON d.driver_id = v.driver_id
        JOIN rfid_cards AS r
            ON r.driver_id = d.driver_id
        WHERE v.driver_id = ?
          AND UPPER(r.uid) = UPPER(?)
          AND d.status = 'ACTIVE'
        ORDER BY v.plate_number
    """, (driver_id, uid)).fetchall()

    return [
        normalize_plate_text(row["plate_number"] or "")
        for row in rows
        if normalize_plate_text(row["plate_number"] or "")
    ]

@plate_camera_bp.route(
    "/api/plate/stream/<gate>/<int:request_id>",
    methods=["GET"],
)
@login_required
def plate_camera_stream(gate, request_id):
    gate = str(gate).strip().upper()

    if gate not in ("ENTRY", "EXIT"):
        return "Cổng camera không hợp lệ.", 400

    # Kiểm tra yêu cầu trước khi mở camera.
    conn = get_db()

    try:
        req = conn.execute("""
            SELECT id, gate_type, state, created_at
            FROM face_auth_requests
            WHERE id = ?
            LIMIT 1
        """, (request_id,)).fetchone()

        queue_head = get_global_pending_request(conn)

        valid_request = (
            req is not None
            and req["state"] == "PENDING"
            and req["gate_type"] == gate
            and time.time() - float(req["created_at"])
                <= FACE_REQUEST_TTL
            and queue_head is not None
            and int(queue_head["id"]) == request_id
            and queue_head["gate_type"] == gate
        )

        if not valid_request:
            return "Yêu cầu không còn đứng đầu FIFO.", 409

    finally:
        conn.close()

    # Dùng CameraWorker chung với chức năng chụp OCR.
    camera = camera_manager.start_camera(gate)

    @stream_with_context
    def generate():
        check_conn = get_db()
        last_check = 0.0

        try:
            while True:
                now = time.monotonic()

                # Kiểm tra FIFO và TTL định kỳ.
                if now - last_check >= 0.5:
                    last_check = now

                    current = check_conn.execute("""
                        SELECT id, gate_type, state, created_at
                        FROM face_auth_requests
                        WHERE id = ?
                        LIMIT 1
                    """, (request_id,)).fetchone()

                    current_head = get_global_pending_request(
                        check_conn
                    )

                    still_valid = (
                        current is not None
                        and current["state"] == "PENDING"
                        and current["gate_type"] == gate
                        and time.time() - float(current["created_at"])
                            <= FACE_REQUEST_TTL
                        and current_head is not None
                        and int(current_head["id"]) == request_id
                        and current_head["gate_type"] == gate
                    )

                    if not still_valid:
                        break

                frame = camera.get_frame()

                if frame is None:
                    if camera.error:
                        current_app.logger.error(
                            "Camera %s: %s",
                            gate,
                            camera.error,
                        )
                        break

                    time.sleep(0.05)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

                time.sleep(0.04)

        except GeneratorExit:
            raise

        except Exception:
            current_app.logger.exception(
                "Lỗi phát camera preview %s",
                gate,
            )

        finally:
            check_conn.close()

            # Hết yêu cầu hoặc trình duyệt đóng preview:
            # giải phóng camera.
            camera_manager.stop_camera(gate)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": (
                "no-store, no-cache, must-revalidate, max-age=0"
            ),
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@plate_camera_bp.route("/api/plate/capture", methods=["POST"])
@login_required
def capture_and_authenticate_plate():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return json_error("Dữ liệu gửi lên không hợp lệ.", 400)

    try:
        request_id = int(data.get("request_id"))
    except (TypeError, ValueError):
        return json_error("Mã yêu cầu không hợp lệ.", 400)

    if request_id <= 0:
        return json_error("Mã yêu cầu không hợp lệ.", 400)

    gate = str(data.get("gate", "")).strip().upper()

    if gate not in ("ENTRY", "EXIT"):
        return json_error("Cổng camera không hợp lệ.", 400)

    conn = get_db()

    try:
        # --------------------------------------------------
        # 1. Kiểm tra yêu cầu RFID, TTL và FIFO trước khi bật camera.
        # --------------------------------------------------
        req = conn.execute("""
            SELECT id, uid, driver_id, gate_type, state, created_at
            FROM face_auth_requests
            WHERE id = ?
            LIMIT 1
        """, (request_id,)).fetchone()

        if not req or req["state"] != "PENDING":
            return json_error("Yêu cầu không còn hiệu lực.", 409)

        if req["gate_type"] != gate:
            return json_error("Yêu cầu thuộc cổng khác.", 403)

        if time.time() - float(req["created_at"]) > FACE_REQUEST_TTL:
            return json_error("Yêu cầu đã hết thời gian xác thực.", 409)

        queue_head = get_global_pending_request(conn)

        if (
            queue_head is None
            or int(queue_head["id"]) != request_id
            or queue_head["gate_type"] != gate
        ):
            return json_error(
                "Yêu cầu chưa đến lượt xử lý theo FIFO chung.",
                409,
            )

        registered_plates = get_registered_plates(
            conn,
            req["driver_id"],
            req["uid"],
        )

        if not registered_plates:
            return json_error(
                "RFID hoặc tài xế không có biển số xe hợp lệ đã đăng ký.",
                403,
            )

        # --------------------------------------------------
        # 2. Chụp một đợt. Hàm tự giải phóng camera sau khi xử lý.
        # --------------------------------------------------
        result = capture_plate_burst(gate, frame_count=5)

        detected_plate = result.get("candidate")
        detected_plate = (
            normalize_plate_text(detected_plate)
            if detected_plate
            else None
        )

        matched_plate = None

        if result.get("success") and detected_plate:
            for registered in registered_plates:
                if detected_plate == registered:
                    matched_plate = registered
                    break

        plate_allowed = matched_plate is not None

        if plate_allowed:
            status = "ALLOW"
            message = "Biển số hợp lệ và khớp với tài xế."

        elif not result.get("success"):
            status = "DENY"
            message = result.get(
                "message",
                "Kết quả nhận diện biển số chưa đủ ổn định.",
            )

        elif not detected_plate:
            status = "DENY"
            message = "Không đọc được biển số."

        else:
            status = "DENY"
            message = (
                "Biển số nhận diện được không thuộc tài xế "
                "đang sử dụng thẻ RFID này."
            )          

        # --------------------------------------------------
        # 3. Kiểm tra lại FIFO, RFID, xe và TTL trước khi ghi DB.
        # --------------------------------------------------
        conn.execute("BEGIN IMMEDIATE")

        current = conn.execute("""
            SELECT id, uid, driver_id, gate_type, state, created_at
            FROM face_auth_requests
            WHERE id = ?
            LIMIT 1
        """, (request_id,)).fetchone()

        current_head = get_global_pending_request(conn)

        if (
            not current
            or current["state"] != "PENDING"
            or current["uid"] != req["uid"]
            or current["driver_id"] != req["driver_id"]
            or current["gate_type"] != gate
            or current_head is None
            or int(current_head["id"]) != request_id
        ):
            conn.rollback()
            return json_error(
                "Yêu cầu đã thay đổi hoặc không còn đứng đầu FIFO.",
                409,
            )

        if time.time() - float(current["created_at"]) > FACE_REQUEST_TTL:
            conn.rollback()
            return json_error(
                "Yêu cầu đã hết thời gian xác thực.",
                409,
            )

        current_registered = get_registered_plates(
            conn,
            current["driver_id"],
            current["uid"],
        )

        if current_registered != registered_plates:
            conn.rollback()
            return json_error(
                "Thông tin RFID hoặc biển số đăng ký đã thay đổi. "
                "Vui lòng quẹt thẻ lại.",
                409,
            )

        conn.execute("""
            INSERT INTO plate_auth_results (
                request_id,
                uid,
                driver_id,
                gate_type,
                detected_plate,
                registered_plate,
                confidence,
                valid_frames,
                matching_frames,
                status,
                message,
                created_at,
                processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(request_id) DO UPDATE SET
                uid = excluded.uid,
                driver_id = excluded.driver_id,
                gate_type = excluded.gate_type,
                detected_plate = excluded.detected_plate,
                registered_plate = excluded.registered_plate,
                confidence = excluded.confidence,
                valid_frames = excluded.valid_frames,
                matching_frames = excluded.matching_frames,
                status = excluded.status,
                message = excluded.message,
                created_at = excluded.created_at,
                processed_at = excluded.processed_at
        """, (
            request_id,
            current["uid"],
            current["driver_id"],
            gate,
            detected_plate,
            matched_plate,
            float(result.get("confidence") or 0.0),
            int(result.get("valid_frames") or 0),
            int(result.get("votes") or 0) if plate_allowed else 0,
            status,
            message,
            time.time(),
            time.time(),
        ))

        conn.commit()

        return jsonify({
            "success": True,
            "plate_allowed": plate_allowed,
            "status": status,
            "request_id": request_id,
            "gate": gate,
            "detected_plate": detected_plate,
            "registered_plate": matched_plate,
            "confidence": float(result.get("confidence") or 0.0),
            "valid_frames": int(result.get("valid_frames") or 0),
            "matching_frames": int(result.get("votes") or 0),
            "message": message,
        })

    except sqlite3.Error:
        conn.rollback()
        current_app.logger.exception(
            "Lỗi database khi xác thực biển số."
        )
        return json_error("Lỗi cơ sở dữ liệu khi xác thực biển số.", 500)

    except Exception:
        conn.rollback()
        current_app.logger.exception(
            "Lỗi khi xác thực biển số."
        )
        return json_error("Không thể xử lý xác thực biển số.", 500)

    finally:
        conn.close()