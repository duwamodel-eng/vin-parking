from pathlib import Path
import base64
import json
import os
import re
import secrets
import sqlite3
import time
from functools import wraps

import cv2
import face_recognition
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "database" / "parking.db"
SECRET_FILE = BASE_DIR / ".flask_secret"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("VINPARKING_SECRET_KEY")
if not app.config["SECRET_KEY"]:
    if SECRET_FILE.exists():
        app.config["SECRET_KEY"] = SECRET_FILE.read_text(encoding="utf-8").strip()
    else:
        app.config["SECRET_KEY"] = secrets.token_hex(32)
        SECRET_FILE.write_text(app.config["SECRET_KEY"], encoding="utf-8")
        SECRET_FILE.chmod(0o600)

app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
# Nếu triển khai qua HTTPS, có thể bật cookie secure bằng biến môi trường.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("VINPARKING_HTTPS", "0") == "1"

EVENT_LABELS = {
    "ENTRY": "Vào bãi thành công",
    "EXIT": "Ra bãi thành công",
    "ENTRY_DENIED": "Từ chối vào",
    "EXIT_DENIED": "Từ chối ra",
}
PENDING_FACES = {}  # Token tạm trong RAM, hết hạn sau 5 phút.


def get_db():
    """Mở kết nối SQLite; không tự tạo lại hoặc xóa cơ sở dữ liệu hiện có."""
    if not DATABASE.exists():
        raise FileNotFoundError(f"Không tìm thấy cơ sở dữ liệu: {DATABASE}")
    conn = sqlite3.connect(str(DATABASE), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_auth():
    """Tạo bảng tài khoản nếu chưa có; chỉ tạo admin khi bảng chưa có tài khoản."""
    if not DATABASE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy database hiện có: {DATABASE}. "
            "Hãy đặt app.py trong thư mục dự án cạnh thư mục database/parking.db."
        )
    conn = sqlite3.connect(str(DATABASE), timeout=10)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS web_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        count = conn.execute("SELECT COUNT(*) FROM web_users").fetchone()[0]
        if count == 0:
            username = os.environ.get("VINPARKING_ADMIN_USER", "admin").strip()
            password = os.environ.get("VINPARKING_ADMIN_PASSWORD", "VinParking@2026!")
            if not username or not password:
                raise RuntimeError("Tài khoản/mật khẩu admin cấu hình không hợp lệ.")
            conn.execute(
                "INSERT INTO web_users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )
        conn.commit()
    finally:
        conn.close()


# Khởi tạo bảng đăng nhập, không thay đổi dữ liệu bãi xe.
init_auth()


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "message": "Chưa đăng nhập."}), 401
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper


def json_error(message, status=400):
    return jsonify({"success": False, "message": message}), status


def normalize_phone(value):
    """Chuẩn hóa số điện thoại Việt Nam về 10 chữ số bắt đầu bằng 0."""
    phone = re.sub(r"[\s().-]", "", str(value or ""))
    if phone.startswith("+84"):
        phone = "0" + phone[3:]
    if not re.fullmatch(r"0\d{9}", phone):
        return None
    return phone


def valid_full_name(value):
    name = str(value or "").strip()
    if not name or len(name) > 100 or re.search(r"\d", name):
        return None
    if not re.search(r"[^\W\d_]", name, re.UNICODE):
        return None
    return name


def cleanup_pending_faces():
    now = time.time()
    for token in [t for t, item in PENDING_FACES.items() if now - item["created_at"] > 300]:
        PENDING_FACES.pop(token, None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = sqlite3.connect(str(DATABASE), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            user = conn.execute(
                "SELECT id, username, password_hash FROM web_users WHERE username = ?",
                (username,),
            ).fetchone()
        finally:
            conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        error = "Tên đăng nhập hoặc mật khẩu không đúng."
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def home():
    return redirect(url_for("dashboard" if "user_id" in session else "login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/history")
@login_required
def history_page():
    return render_template("history.html")


@app.route("/drivers")
@login_required
def drivers_page():
    return render_template("drivers.html")


@app.route("/api/status")
@login_required
def status():
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT slot_id, status, updated_at FROM parking_slots ORDER BY slot_id"
        ).fetchall()
    finally:
        conn.close()
    slots = [{"slot_id": r["slot_id"], "status": r["status"], "updated_at": r["updated_at"]} for r in rows]
    occupied = sum(1 for slot in slots if slot["status"] == 0)
    return jsonify({"total": len(slots), "occupied": occupied, "free": len(slots) - occupied, "slots": slots})


@app.route("/api/history")
@login_required
def history():
    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(100, max(1, int(request.args.get("limit", 20))))
    except (TypeError, ValueError):
        return json_error("page/limit không hợp lệ.")

    event_type = request.args.get("event_type", "").strip().upper()
    if event_type and event_type not in EVENT_LABELS:
        return json_error("Loại sự kiện không hợp lệ.")

    where = "WHERE h.event_type = ?" if event_type else ""
    params = [event_type] if event_type else []
    conn = get_db()
    try:
        total = conn.execute(
            f"SELECT COUNT(*) FROM parking_history AS h {where}", params
        ).fetchone()[0]
        rows = conn.execute(f"""
            SELECT h.id, h.event_type, h.slot_id, h.uid, h.created_at,
                   d.driver_id, d.full_name
            FROM parking_history AS h
            LEFT JOIN rfid_cards AS r ON UPPER(r.uid) = UPPER(h.uid)
            LEFT JOIN drivers AS d ON d.driver_id = r.driver_id
            {where}
            ORDER BY h.id DESC
            LIMIT ? OFFSET ?
        """, params + [limit, (page - 1) * limit]).fetchall()
    finally:
        conn.close()

    events = [{
        "id": row["id"],
        "event_type": row["event_type"],
        "label": EVENT_LABELS.get(row["event_type"], row["event_type"]),
        "slot_id": row["slot_id"],
        "uid": row["uid"],
        "created_at": row["created_at"],
        "driver_id": row["driver_id"],
        "full_name": row["full_name"],
    } for row in rows]
    return jsonify({"events": events, "total": total, "page": page, "limit": limit})


@app.route("/api/drivers")
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
            "driver_id": r["driver_id"], "full_name": r["full_name"],
            "phone": r["phone"], "status": r["status"],
            "created_at": r["created_at"], "uid": r["uid"],
            "face_registered": bool(r["face_registered"]),
        } for r in rows]
        return jsonify({"success": True, "drivers": drivers, "total": len(drivers)})
    except sqlite3.Error:
        app.logger.exception("Lỗi truy vấn danh sách tài xế")
        return json_error("Không thể tải danh sách tài xế.", 500)
    finally:
        conn.close()


@app.route("/api/face/encode", methods=["POST"])
@login_required
def encode_driver_face():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("Dữ liệu ảnh không hợp lệ.")
    image_data = data.get("image", "")
    if not isinstance(image_data, str):
        return json_error("Ảnh không đúng định dạng JPEG.")
    match = re.fullmatch(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", image_data)
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
            bgr = cv2.resize(bgr, (max(1, int(width * scale)), max(1, int(height * scale))))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if len(locations) != 1:
            return json_error(f"Cần đúng một khuôn mặt trong ảnh. Hiện phát hiện: {len(locations)}.")
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
        if len(encodings) != 1 or not np.isfinite(encodings[0]).all():
            return json_error("Không tạo được dữ liệu khuôn mặt hợp lệ.")
        cleanup_pending_faces()
        token = secrets.token_urlsafe(32)
        PENDING_FACES[token] = {
            "encoding": encodings[0].tolist(),
            "created_at": time.time(),
            "user_id": session["user_id"],
        }
        return jsonify({"success": True, "face_token": token, "message": "Đã tạo dữ liệu khuôn mặt."})
    except (ValueError, cv2.error):
        return json_error("Ảnh không hợp lệ hoặc không thể xử lý.")


@app.route("/api/drivers/register", methods=["POST"])
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
        return json_error("Họ tên không hợp lệ; không để trống, không chứa số và tối đa 100 ký tự.")
    if phone is None:
        return json_error("Số điện thoại phải có 10 chữ số, bắt đầu bằng 0.")
    if not re.fullmatch(r"(?:[0-9A-F]{2}:)+[0-9A-F]{2}", uid):
        return json_error("UID RFID không đúng định dạng.")

    cleanup_pending_faces()
    face = PENDING_FACES.get(face_token)
    if not face or face["user_id"] != session["user_id"] or time.time() - face["created_at"] > 300:
        return json_error("Khuôn mặt chưa được xử lý hoặc token đã hết hạn. Hãy chụp lại.")
    encoding = face.get("encoding")
    if not isinstance(encoding, list) or len(encoding) != 128 or not np.isfinite(encoding).all():
        return json_error("Dữ liệu khuôn mặt không hợp lệ.")

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
                # Kiem tra UID da duoc dang ky chua
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
                # Chan dang ky lai khuon mat da ton tai
        existing_faces = conn.execute(
            "SELECT driver_id, encoding_json FROM face_encodings"
        ).fetchall()

        new_encoding = np.asarray(encoding, dtype=np.float64)

        for existing_driver_id, encoding_json in existing_faces:
            try:
                known_encoding = np.asarray(
                    json.loads(encoding_json), dtype=np.float64
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
            "INSERT INTO drivers (driver_id, full_name, phone, status) VALUES (?, ?, ?, 'ACTIVE')",
            (driver_id, full_name, phone),
        )
        conn.execute("INSERT INTO rfid_cards (uid, driver_id) VALUES (?, ?)", (uid, driver_id))
        conn.execute("""
            INSERT INTO face_encodings (driver_id, encoding_json, samples, updated_at)
            VALUES (?, ?, 1, datetime('now', 'localtime'))
        """, (driver_id, json.dumps(encoding)))
        conn.commit()
        PENDING_FACES.pop(face_token, None)
        return jsonify({"success": True, "message": "Đăng ký tài xế thành công."}), 201
    except sqlite3.IntegrityError:
        conn.rollback()
        return json_error("Mã tài xế, UID hoặc hồ sơ khuôn mặt đã tồn tại.", 409)
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Lỗi lưu hồ sơ tài xế")
        return json_error("Lỗi cơ sở dữ liệu. Hồ sơ chưa được lưu.", 500)
    finally:
        conn.close()


@app.route("/api/drivers/<driver_id>", methods=["PUT"])
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
            "UPDATE drivers SET full_name = ?, phone = ?, status = ? WHERE driver_id = ?",
            (full_name, phone, status_value, driver_id),
        )
        if cur.rowcount == 0:
            conn.rollback()
            return json_error("Không tìm thấy tài xế.", 404)
        conn.commit()
        return jsonify({"success": True, "message": "Cập nhật tài xế thành công."})
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Lỗi cập nhật tài xế")
        return json_error("Không thể cập nhật tài xế.", 500)
    finally:
        conn.close()


@app.route("/api/drivers/<driver_id>", methods=["DELETE"])
@login_required
def delete_driver(driver_id):
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        exists = conn.execute("SELECT 1 FROM drivers WHERE driver_id = ?", (driver_id,)).fetchone()
        if exists is None:
            conn.rollback()
            return json_error("Không tìm thấy tài xế.", 404)
        # Chỉ xóa hồ sơ hiện hành; không xóa parking_history.
        conn.execute("DELETE FROM face_encodings WHERE driver_id = ?", (driver_id,))
        conn.execute("DELETE FROM rfid_cards WHERE driver_id = ?", (driver_id,))
        conn.execute("DELETE FROM drivers WHERE driver_id = ?", (driver_id,))
        conn.commit()
        return jsonify({"success": True, "message": "Đã xóa hồ sơ tài xế. Lịch sử bãi xe được giữ nguyên."})
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Lỗi xóa tài xế")
        return json_error("Không thể xóa tài xế.", 500)
    finally:
        conn.close()


@app.route("/api/rfid/scan/start", methods=["POST"])
@login_required
def start_rfid_scan():
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rfid_scan_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active_until REAL NOT NULL DEFAULT 0
            )
        """)
        conn.execute("INSERT OR IGNORE INTO rfid_scan_state (id, active_until) VALUES (1, 0)")
        # Xóa UID đăng ký còn tồn từ phiên cũ, không đụng đến các nguồn khác.
        conn.execute("DELETE FROM rfid_scan_queue WHERE source = 'REGISTRATION'")
        conn.execute("UPDATE rfid_scan_state SET active_until = ? WHERE id = 1", (time.time() + 30,))
        conn.commit()
        return jsonify({"success": True, "message": "Đưa thẻ vào đầu đọc cổng vào trong 30 giây."})
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Không thể bắt đầu phiên quét RFID")
        return json_error("Không thể bắt đầu quét RFID.", 500)
    finally:
        conn.close()


@app.route("/api/rfid/scan/result", methods=["GET"])
@login_required
def get_rfid_scan_result():
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rfid_scan_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active_until REAL NOT NULL DEFAULT 0
            )
        """)
        conn.execute("INSERT OR IGNORE INTO rfid_scan_state (id, active_until) VALUES (1, 0)")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("""
            SELECT id, uid FROM rfid_scan_queue
            WHERE source = 'REGISTRATION' AND claimed = 0
            ORDER BY id ASC LIMIT 1
        """).fetchone()
        if row:
            conn.execute("UPDATE rfid_scan_queue SET claimed = 1 WHERE id = ?", (row["id"],))
            conn.commit()
            return jsonify({"success": True, "found": True, "uid": row["uid"]})
        state = conn.execute("SELECT active_until FROM rfid_scan_state WHERE id = 1").fetchone()
        conn.commit()
        active = bool(state and state["active_until"] > time.time())
        return jsonify({
            "success": True, "found": False, "active": active,
            "message": "Đang chờ quét thẻ." if active else "Phiên quét đã hết hạn hoặc chưa bắt đầu.",
        })
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Không thể đọc kết quả quét RFID")
        return json_error("Không thể đọc kết quả RFID.", 500)
    finally:
        conn.close()


@app.route("/api/parking/update", methods=["POST"])
@login_required
def update_parking():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("JSON không hợp lệ.")
    required = ("P1", "P2", "P3", "P4")
    for slot in required:
        if slot not in data or type(data[slot]) is not int or data[slot] not in (0, 1):
            return json_error(f"{slot} phải là số nguyên 0 hoặc 1.")

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for slot in required:
            cur = conn.execute(
                "UPDATE parking_slots SET status = ?, updated_at = datetime('now', 'localtime') WHERE slot_id = ?",
                (data[slot], slot),
            )
            if cur.rowcount == 0:
                conn.rollback()
                return json_error(f"Không tìm thấy vị trí {slot} trong cơ sở dữ liệu.", 404)
        conn.commit()
        return jsonify({"success": True, "message": "Cập nhật thành công", **{slot: data[slot] for slot in required}})
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Lỗi cập nhật trạng thái bãi xe")
        return json_error("Không thể cập nhật trạng thái bãi xe.", 500)
    finally:
        conn.close()


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return json_error("Không tìm thấy API.", 404)
    return "Không tìm thấy trang.", 404


@app.errorhandler(500)
def internal_error(_error):
    if request.path.startswith("/api/"):
        return json_error("Lỗi máy chủ nội bộ.", 500)
    return "Đã xảy ra lỗi máy chủ.", 500


if __name__ == "__main__":
    # Chỉ dùng debug=False trên Raspberry Pi; chạy qua mạng LAN theo nhu cầu.
    app.run(host="0.0.0.0", port=5000, debug=False)
