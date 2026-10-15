import os
import secrets
import sqlite3
import time

from flask import Flask, jsonify, request
from werkzeug.security import generate_password_hash

from services.app_context import BASE_DIR, DATABASE, get_db, json_error
from services.auth_utils import login_required


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

def init_face_auth():
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_auth_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL,
                driver_id TEXT,
                state TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (state IN ('PENDING', 'ALLOW', 'DENY', 'EXPIRED')),
                created_at REAL NOT NULL,
                processed_at REAL,
                message TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_auth_pending
            ON face_auth_requests(state, created_at)
        """)
        conn.commit()
    finally:
        conn.close()


init_face_auth()


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

from routes.page_routes import page_bp
from routes.auth_routes import auth_bp
from routes.driver_routes import driver_bp
from routes.face_routes import face_bp
from routes.parking_routes import parking_bp

app.register_blueprint(auth_bp)
app.register_blueprint(page_bp)
app.register_blueprint(driver_bp)
app.register_blueprint(face_bp)
app.register_blueprint(parking_bp)

if __name__ == "__main__":
    # Chỉ dùng debug=False trên Raspberry Pi; chạy qua mạng LAN theo nhu cầu.
    app.run(host="0.0.0.0", port=5000, debug=False)
